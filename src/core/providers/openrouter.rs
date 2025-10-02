//! OpenRouter provider implementation

use crate::core::provider::{Provider, ProviderError};
use crate::models::openai::{
    OpenAIChatCompletionRequest, OpenAIChatCompletionResponse, OpenAIStreamOptions,
};
use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::Stream;
use reqwest::Client;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, Notify};
use tracing::{error, warn};

/// OpenRouter provider
pub struct OpenRouterProvider {
    client: Client,
    api_key: String,
    base_url: String,
    site_url: Option<String>,
    app_name: Option<String>,
    active_requests: Arc<Mutex<HashMap<String, Arc<Notify>>>>,
}

impl OpenRouterProvider {
    /// Create a new OpenRouter provider
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenRouter API key
    /// * `base_url` - OpenRouter API base URL (default: https://openrouter.ai/api/v1)
    /// * `timeout` - Request timeout in seconds
    /// * `site_url` - Optional site URL for OpenRouter credits
    /// * `app_name` - Optional application name
    pub fn new(
        api_key: String,
        base_url: Option<String>,
        timeout: u64,
        site_url: Option<String>,
        app_name: Option<String>,
    ) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            api_key,
            base_url: base_url.unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string()),
            site_url,
            app_name,
            active_requests: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Classify OpenRouter errors and provide helpful messages
    fn classify_error(error_detail: &str) -> String {
        let error_lower = error_detail.to_lowercase();

        if error_lower.contains("invalid") && error_lower.contains("api") {
            return "Invalid API key. Please check your OPENROUTER_API_KEY configuration."
                .to_string();
        }

        if error_lower.contains("rate_limit") || error_lower.contains("quota") {
            return "Rate limit exceeded. Please wait and try again.".to_string();
        }

        if error_lower.contains("insufficient") && error_lower.contains("credits") {
            return "Insufficient credits. Please add credits to your OpenRouter account."
                .to_string();
        }

        if error_lower.contains("model")
            && (error_lower.contains("not found") || error_lower.contains("does not exist"))
        {
            return "Model not found. Please check your model configuration.".to_string();
        }

        error_detail.to_string()
    }

    /// Internal method to send completion request
    async fn send_completion_request(
        &self,
        request: &OpenAIChatCompletionRequest,
        _cancel_notify: Option<Arc<Notify>>,
    ) -> Result<OpenAIChatCompletionResponse, ProviderError> {
        let url = format!("{}/chat/completions", self.base_url);

        // Log outgoing request to OpenRouter
        tracing::info!(
            "Sending request to OpenRouter: model={}, messages={}, max_tokens={:?}, clamped_max_tokens={}",
            request.model,
            request.messages.len(),
            request.max_tokens,
            request.max_tokens.unwrap_or(0)
        );

        // Debug log to show message content for token count investigation
        if request.messages.len() > 10 {
            let total_content_len: usize = request.messages
                .iter()
                .map(|msg| {
                    if let Some(content) = &msg.content {
                        match content {
                            serde_json::Value::String(s) => s.len(),
                            serde_json::Value::Array(arr) => arr.iter()
                                .filter_map(|v| v.get("text").and_then(|t| t.as_str()))
                                .map(|s| s.len())
                                .sum::<usize>(),
                            _ => 0,
                        }
                    } else {
                        0
                    }
                })
                .sum();

            tracing::info!(
                "Message debug: total_raw_content_chars={}, avg_chars_per_msg={}",
                total_content_len,
                total_content_len / request.messages.len()
            );
        }

        let mut req_builder = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .bearer_auth(&self.api_key);

        // Add OpenRouter-specific headers
        if let Some(ref site_url) = self.site_url {
            req_builder = req_builder.header("HTTP-Referer", site_url);
        }
        if let Some(ref app_name) = self.app_name {
            req_builder = req_builder.header("X-Title", app_name);
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
                402 => ProviderError::BadRequest(
                    "Insufficient credits. Please add credits to your OpenRouter account."
                        .to_string(),
                ),
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

        // Log token usage information
        tracing::info!(
            "OpenRouter response: model={}, sent_tokens={}, received_tokens={}, total_tokens={}",
            completion.model,
            completion.usage.prompt_tokens,
            completion.usage.completion_tokens,
            completion.usage.total_tokens
        );

        Ok(completion)
    }

    /// Internal method to send streaming request
    async fn send_stream_request(
        &self,
        request: &OpenAIChatCompletionRequest,
    ) -> Result<reqwest::Response, ProviderError> {
        let url = format!("{}/chat/completions", self.base_url);

        // Log outgoing streaming request to OpenRouter
        tracing::info!(
            "Sending STREAM request to OpenRouter: model={}, messages={}, max_tokens={:?}, clamped_max_tokens={}",
            request.model,
            request.messages.len(),
            request.max_tokens,
            request.max_tokens.unwrap_or(0)
        );

        let mut req_builder = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .bearer_auth(&self.api_key);

        // Add OpenRouter-specific headers
        if let Some(ref site_url) = self.site_url {
            req_builder = req_builder.header("HTTP-Referer", site_url);
        }
        if let Some(ref app_name) = self.app_name {
            req_builder = req_builder.header("X-Title", app_name);
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
                402 => ProviderError::BadRequest(
                    "Insufficient credits. Please add credits to your OpenRouter account."
                        .to_string(),
                ),
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
impl Provider for OpenRouterProvider {
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
        "OpenRouter"
    }
}
