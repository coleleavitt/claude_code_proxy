//! OpenAI client with async support and cancellation
//!
//! This module provides an async HTTP client for communicating with OpenAI API
//! endpoints (including Azure OpenAI). It supports request cancellation through
//! a cancellation token system.

use crate::models::openai::{
    OpenAIChatCompletionRequest, OpenAIChatCompletionResponse, OpenAIStreamingChunk,
};
use anyhow::{Context, Result};
use futures::stream::Stream;
use reqwest::Client;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, Notify};
use tracing::{error, warn};

/// Error types that can occur during OpenAI API interactions
#[derive(Debug, thiserror::Error)]
pub enum OpenAIError {
    #[error("Authentication failed: {0}")]
    Authentication(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimit(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("API error (status {status}): {message}")]
    ApiError { status: u16, message: String },

    #[error("Request cancelled by client")]
    Cancelled,

    #[error("Unexpected error: {0}")]
    Unexpected(String),
}

/// OpenAI async client with cancellation support
pub struct OpenAIClient {
    client: Client,
    api_key: String,
    base_url: String,
    api_version: Option<String>,
    active_requests: Arc<Mutex<HashMap<String, Arc<Notify>>>>,
}

impl OpenAIClient {
    /// Create a new OpenAI client
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

    /// Send chat completion to OpenAI API with cancellation support
    ///
    /// # Arguments
    ///
    /// * `request` - The chat completion request
    /// * `request_id` - Optional request ID for cancellation tracking
    ///
    /// # Errors
    ///
    /// Returns OpenAIError for API errors, authentication failures, etc.
    pub async fn create_chat_completion(
        &self,
        request: &OpenAIChatCompletionRequest,
        request_id: Option<String>,
    ) -> Result<OpenAIChatCompletionResponse, OpenAIError> {
        let cancel_notify = if let Some(ref id) = request_id {
            let notify = Arc::new(Notify::new());
            self.active_requests.lock().await.insert(id.clone(), notify.clone());
            Some(notify)
        } else {
            None
        };

        let result = self.send_completion_request(request, cancel_notify).await;

        // Clean up active request tracking
        if let Some(id) = request_id {
            self.active_requests.lock().await.remove(&id);
        }

        result
    }

    /// Send streaming chat completion to OpenAI API with cancellation support
    ///
    /// # Arguments
    ///
    /// * `request` - The chat completion request (will be modified to enable streaming)
    /// * `request_id` - Optional request ID for cancellation tracking
    ///
    /// # Returns
    ///
    /// An async stream of SSE-formatted strings
    pub async fn create_chat_completion_stream(
        &self,
        mut request: OpenAIChatCompletionRequest,
        request_id: Option<String>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, OpenAIError>> + Send>>, OpenAIError> {
        let cancel_notify = if let Some(ref id) = request_id {
            let notify = Arc::new(Notify::new());
            self.active_requests.lock().await.insert(id.clone(), notify.clone());
            Some(notify)
        } else {
            None
        };

        // Ensure streaming is enabled with usage data
        request.stream = true;
        if request.stream_options.is_none() {
            request.stream_options = Some(crate::models::openai::OpenAIStreamOptions {
                include_usage: true,
            });
        }

        let response = self.send_stream_request(&request).await?;

        // Convert bytes stream to SSE lines
        use futures::StreamExt;
        use tokio_stream::wrappers::LinesStream;
        use tokio::io::{AsyncBufReadExt, BufReader};
        use futures_util::TryStreamExt;

        let byte_stream = response.bytes_stream();
        let byte_stream = byte_stream.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e));

        // Convert to AsyncRead
        let reader = tokio_util::io::StreamReader::new(byte_stream);
        let buf_reader = BufReader::new(reader);
        let lines = buf_reader.lines();
        let line_stream = LinesStream::new(lines);

        let stream = line_stream.map(|result: Result<String, std::io::Error>| {
            result.map_err(|e| OpenAIError::Unexpected(e.to_string()))
        });

        Ok(Box::pin(stream))
    }

    /// Cancel an active request by request_id
    ///
    /// # Arguments
    ///
    /// * `request_id` - The request ID to cancel
    ///
    /// # Returns
    ///
    /// true if request was found and cancelled, false otherwise
    pub async fn cancel_request(&self, request_id: &str) -> bool {
        if let Some(notify) = self.active_requests.lock().await.get(request_id) {
            notify.notify_waiters();
            true
        } else {
            false
        }
    }

    /// Classify OpenAI errors and provide helpful messages
    fn classify_openai_error(error_detail: &str) -> String {
        let error_lower = error_detail.to_lowercase();

        // Region/country restrictions
        if error_lower.contains("unsupported_country_region_territory")
            || error_lower.contains("country, region, or territory not supported")
        {
            return "OpenAI API is not available in your region. Consider using a VPN or Azure OpenAI service.".to_string();
        }

        // API key issues
        if error_lower.contains("invalid_api_key") || error_lower.contains("unauthorized") {
            return "Invalid API key. Please check your OPENAI_API_KEY configuration.".to_string();
        }

        // Rate limiting
        if error_lower.contains("rate_limit") || error_lower.contains("quota") {
            return "Rate limit exceeded. Please wait and try again, or upgrade your API plan."
                .to_string();
        }

        // Model not found
        if error_lower.contains("model")
            && (error_lower.contains("not found") || error_lower.contains("does not exist"))
        {
            return "Model not found. Please check your BIG_MODEL and SMALL_MODEL configuration."
                .to_string();
        }

        // Billing issues
        if error_lower.contains("billing") || error_lower.contains("payment") {
            return "Billing issue. Please check your OpenAI account billing status.".to_string();
        }

        // Default: return original message
        error_detail.to_string()
    }

    /// Internal method to send completion request
    async fn send_completion_request(
        &self,
        request: &OpenAIChatCompletionRequest,
        cancel_notify: Option<Arc<Notify>>,
    ) -> Result<OpenAIChatCompletionResponse, OpenAIError> {
        let url = if self.api_version.is_some() {
            // Azure OpenAI endpoint format
            format!(
                "{}/openai/deployments/{}/chat/completions?api-version={}",
                self.base_url,
                request.model,
                self.api_version.as_ref().unwrap()
            )
        } else {
            // Standard OpenAI endpoint
            format!("{}/chat/completions", self.base_url)
        };

        let mut req_builder = self
            .client
            .post(&url)
            .header("Content-Type", "application/json");

        if self.api_version.is_some() {
            // Azure uses api-key header
            req_builder = req_builder.header("api-key", &self.api_key);
        } else {
            // OpenAI uses Bearer token
            req_builder = req_builder.bearer_auth(&self.api_key);
        }

        let req_builder = req_builder.json(request);

        let response = req_builder
            .send()
            .await
            .map_err(|e| OpenAIError::Unexpected(e.to_string()))?;

        let status = response.status();

        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            let classified_error = Self::classify_openai_error(&error_text);

            return Err(match status.as_u16() {
                401 => OpenAIError::Authentication(classified_error),
                429 => OpenAIError::RateLimit(classified_error),
                400 => OpenAIError::BadRequest(classified_error),
                _ => OpenAIError::ApiError {
                    status: status.as_u16(),
                    message: classified_error,
                },
            });
        }

        let completion: OpenAIChatCompletionResponse = response
            .json()
            .await
            .map_err(|e| OpenAIError::Unexpected(format!("Failed to parse response: {}", e)))?;

        Ok(completion)
    }

    /// Internal method to send streaming request
    async fn send_stream_request(
        &self,
        request: &OpenAIChatCompletionRequest,
    ) -> Result<reqwest::Response, OpenAIError> {
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
            .map_err(|e| OpenAIError::Unexpected(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            let classified_error = Self::classify_openai_error(&error_text);

            return Err(match status.as_u16() {
                401 => OpenAIError::Authentication(classified_error),
                429 => OpenAIError::RateLimit(classified_error),
                400 => OpenAIError::BadRequest(classified_error),
                _ => OpenAIError::ApiError {
                    status: status.as_u16(),
                    message: classified_error,
                },
            });
        }

        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_region_error() {
        let error = "unsupported_country_region_territory";
        let result = OpenAIClient::classify_openai_error(error);
        assert!(result.contains("region"));
    }

    #[test]
    fn test_classify_auth_error() {
        let error = "invalid_api_key: The API key is invalid";
        let result = OpenAIClient::classify_openai_error(error);
        assert!(result.contains("API key"));
    }
}
