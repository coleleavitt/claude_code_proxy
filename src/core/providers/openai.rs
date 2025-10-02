//! OpenAI provider implementation

use crate::core::provider::{Provider, ProviderError};
use crate::models::openai::{
    OpenAIChatCompletionRequest, OpenAIChatCompletionResponse, OpenAIStreamOptions,
};
use async_trait::async_trait;
use futures::stream::Stream;
use futures::StreamExt;
use reqwest::Client;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, Notify};
use tracing::{error, warn};

/// OpenAI provider (supports OpenAI and Azure OpenAI)
pub struct OpenAIProvider {
    client: Client,
    api_key: String,
    base_url: String,
    api_version: Option<String>,
    active_requests: Arc<Mutex<HashMap<String, Arc<Notify>>>>,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenAI API key
    /// * `base_url` - OpenAI API base URL or Azure endpoint
    /// * `timeout` - Request timeout in seconds
    /// * `api_version` - Optional Azure API version (enables Azure mode)
    pub fn new(
        api_key: String,
        base_url: String,
        timeout: u64,
        api_version: Option<String>,
    ) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            api_key,
            base_url,
            api_version,
            active_requests: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Classify OpenAI errors and provide helpful messages
    fn classify_error(error_detail: &str) -> String {
        let error_lower = error_detail.to_lowercase();

        if error_lower.contains("unsupported_country_region_territory")
            || error_lower.contains("country, region, or territory not supported")
        {
            return "OpenAI API is not available in your region. Consider using a VPN or Azure OpenAI service.".to_string();
        }

        if error_lower.contains("invalid_api_key") || error_lower.contains("unauthorized") {
            return "Invalid API key. Please check your OPENAI_API_KEY configuration.".to_string();
        }

        if error_lower.contains("rate_limit") || error_lower.contains("quota") {
            return "Rate limit exceeded. Please wait and try again, or upgrade your API plan."
                .to_string();
        }

        if error_lower.contains("model")
            && (error_lower.contains("not found") || error_lower.contains("does not exist"))
        {
            return "Model not found. Please check your model configuration.".to_string();
        }

        if error_lower.contains("billing") || error_lower.contains("payment") {
            return "Billing issue. Please check your OpenAI account billing status.".to_string();
        }

        error_detail.to_string()
    }

    /// Internal method to send completion request
    async fn send_completion_request(
        &self,
        request: &OpenAIChatCompletionRequest,
        _cancel_notify: Option<Arc<Notify>>,
    ) -> Result<OpenAIChatCompletionResponse, ProviderError> {
        let url = if self.api_version.is_some() {
            format!(
                "{}/openai/deployments/{}/chat/completions?api-version={}",
                self.base_url,
                request.model,
                self.api_version.as_ref().unwrap()
            )
        } else {
            format!("{}/chat/completions", self.base_url)
        };

        let mut req_builder = self
            .client
            .post(&url)
            .header("Content-Type", "application/json");

        if self.api_version.is_some() {
            req_builder = req_builder.header("api-key", &self.api_key);
        } else {
            req_builder = req_builder.bearer_auth(&self.api_key);
        }

        let req_builder = req_builder.json(request);

        let response = req_builder
            .send()
            .await
            .map_err(|e| ProviderError::Unexpected(e.to_string()))?;

        let status = response.status();

        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            let classified_error = Self::classify_error(&error_text);

            return Err(match status.as_u16() {
                401 => ProviderError::Authentication(classified_error),
                429 => ProviderError::RateLimit(classified_error),
                400 => ProviderError::BadRequest(classified_error),
                _ => ProviderError::ApiError {
                    status: status.as_u16(),
                    message: classified_error,
                },
            });
        }

        let completion: OpenAIChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::Unexpected(format!("Failed to parse response: {}", e)))?;

        Ok(completion)
    }

    /// Internal method to send streaming request
    async fn send_stream_request(
        &self,
        request: &OpenAIChatCompletionRequest,
    ) -> Result<reqwest::Response, ProviderError> {
        let url = if self.api_version.is_some() {
            format!(
                "{}/openai/deployments/{}/chat/completions?api-version={}",
                self.base_url,
                request.model,
                self.api_version.as_ref().unwrap()
            )
        } else {
            format!("{}/chat/completions", self.base_url)
        };

        let mut req_builder = self
            .client
            .post(&url)
            .header("Content-Type", "application/json");

        if self.api_version.is_some() {
            req_builder = req_builder.header("api-key", &self.api_key);
        } else {
            req_builder = req_builder.bearer_auth(&self.api_key);
        }

        let response = req_builder
            .json(request)
            .send()
            .await
            .map_err(|e| ProviderError::Unexpected(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            let classified_error = Self::classify_error(&error_text);

            return Err(match status.as_u16() {
                401 => ProviderError::Authentication(classified_error),
                429 => ProviderError::RateLimit(classified_error),
                400 => ProviderError::BadRequest(classified_error),
                _ => ProviderError::ApiError {
                    status: status.as_u16(),
                    message: classified_error,
                },
            });
        }

        Ok(response)
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    async fn create_chat_completion(
        &self,
        request: &OpenAIChatCompletionRequest,
        request_id: Option<String>,
    ) -> Result<OpenAIChatCompletionResponse, ProviderError> {
        let cancel_notify = if let Some(ref id) = request_id {
            let notify = Arc::new(Notify::new());
            self.active_requests
                .lock()
                .await
                .insert(id.clone(), notify.clone());
            Some(notify)
        } else {
            None
        };

        let result = self.send_completion_request(request, cancel_notify).await;

        if let Some(id) = request_id {
            self.active_requests.lock().await.remove(&id);
        }

        result
    }

    async fn create_chat_completion_stream(
        &self,
        mut request: OpenAIChatCompletionRequest,
        request_id: Option<String>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, ProviderError>> + Send>>, ProviderError>
    {
        let cancel_notify = if let Some(ref id) = request_id {
            let notify = Arc::new(Notify::new());
            self.active_requests
                .lock()
                .await
                .insert(id.clone(), notify.clone());
            Some(notify)
        } else {
            None
        };

        request.stream = true;
        if request.stream_options.is_none() {
            request.stream_options = Some(OpenAIStreamOptions {
                include_usage: true,
            });
        }

        let response = self.send_stream_request(&request).await?;

        use futures::TryStreamExt;
        use tokio::io::AsyncBufReadExt;
        use tokio_stream::wrappers::LinesStream;

        let byte_stream = response.bytes_stream();
        let byte_stream =
            byte_stream.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));

        let reader = tokio_util::io::StreamReader::new(byte_stream);
        let buf_reader = tokio::io::BufReader::new(reader);
        let lines = buf_reader.lines();
        let line_stream = LinesStream::new(lines);

        let stream = line_stream.map(|result: Result<String, std::io::Error>| {
            result.map_err(|e| ProviderError::Unexpected(e.to_string()))
        });

        Ok(Box::pin(stream))
    }

    async fn cancel_request(&self, request_id: &str) -> bool {
        if let Some(notify) = self.active_requests.lock().await.get(request_id) {
            notify.notify_waiters();
            true
        } else {
            false
        }
    }

    fn provider_name(&self) -> &str {
        if self.api_version.is_some() {
            "Azure OpenAI"
        } else {
            "OpenAI"
        }
    }
}
