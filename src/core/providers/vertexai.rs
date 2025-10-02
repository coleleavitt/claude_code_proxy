//! Vertex AI provider implementation

use crate::core::provider::{Provider, ProviderError};
use crate::models::openai::{
    OpenAIChatCompletionRequest, OpenAIChatCompletionResponse, OpenAIMessage, OpenAIStreamOptions,
    OpenAIChoice, OpenAIUsage,
};
use async_trait::async_trait;
use futures::stream::Stream;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, Notify};
use tracing::{debug, error, warn};

/// Vertex AI provider for Google Cloud's Gemini models
pub struct VertexAIProvider {
    client: Client,
    project_id: String,
    location: String,
    access_token: String,
    active_requests: Arc<Mutex<HashMap<String, Arc<Notify>>>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VertexAIRequest {
    contents: Vec<VertexAIContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<VertexAIGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safety_settings: Option<Vec<VertexAISafetySetting>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VertexAIContent {
    role: String,
    parts: Vec<VertexAIPart>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VertexAIPart {
    text: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct VertexAIGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VertexAISafetySetting {
    category: String,
    threshold: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct VertexAIResponse {
    candidates: Vec<VertexAICandidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage_metadata: Option<VertexAIUsageMetadata>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VertexAICandidate {
    content: VertexAIContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexAIUsageMetadata {
    prompt_token_count: u32,
    candidates_token_count: u32,
    total_token_count: u32,
}

impl VertexAIProvider {
    /// Create a new Vertex AI provider
    ///
    /// # Arguments
    ///
    /// * `project_id` - Google Cloud project ID
    /// * `location` - Google Cloud location (e.g., "us-central1")
    /// * `access_token` - Google Cloud access token (from gcloud auth or service account)
    /// * `timeout` - Request timeout in seconds
    pub fn new(
        project_id: String,
        location: String,
        access_token: String,
        timeout: u64,
    ) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            project_id,
            location,
            access_token,
            active_requests: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Convert OpenAI request to Vertex AI format
    fn convert_request_to_vertex(&self, request: &OpenAIChatCompletionRequest) -> VertexAIRequest {
        let mut contents = Vec::new();

        for msg in &request.messages {
            let role = match msg.role.as_str() {
                "system" => "user", // Vertex AI doesn't have system role, merge with user
                "assistant" => "model",
                _ => "user",
            };

            if let Some(ref content) = msg.content {
                let text = match content {
                    serde_json::Value::String(s) => s.clone(),
                    serde_json::Value::Array(arr) => {
                        // Handle array of content blocks
                        arr.iter()
                            .filter_map(|v| v.get("text").and_then(|t| t.as_str()))
                            .collect::<Vec<_>>()
                            .join("\n")
                    }
                    _ => content.to_string(),
                };

                contents.push(VertexAIContent {
                    role: role.to_string(),
                    parts: vec![VertexAIPart { text }],
                });
            }
        }

        let generation_config = Some(VertexAIGenerationConfig {
            temperature: request.temperature,
            top_p: request.top_p,
            max_output_tokens: request.max_tokens,
        });

        VertexAIRequest {
            contents,
            generation_config,
            safety_settings: None,
        }
    }

    /// Convert Vertex AI response to OpenAI format
    fn convert_response_from_vertex(
        &self,
        response: VertexAIResponse,
        model: &str,
    ) -> OpenAIChatCompletionResponse {
        let choice = if let Some(candidate) = response.candidates.first() {
            let content = candidate
                .content
                .parts
                .iter()
                .map(|p| p.text.clone())
                .collect::<Vec<_>>()
                .join("\n");

            let finish_reason = candidate
                .finish_reason
                .as_ref()
                .map(|r| match r.as_str() {
                    "STOP" => "stop",
                    "MAX_TOKENS" => "length",
                    _ => "stop",
                })
                .unwrap_or("stop")
                .to_string();

            OpenAIChoice {
                index: 0,
                message: OpenAIMessage {
                    role: "assistant".to_string(),
                    content: Some(serde_json::Value::String(content)),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some(finish_reason),
            }
        } else {
            OpenAIChoice {
                index: 0,
                message: OpenAIMessage {
                    role: "assistant".to_string(),
                    content: Some(serde_json::Value::String("".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: Some("stop".to_string()),
            }
        };

        let usage = response.usage_metadata.map(|u| OpenAIUsage {
            prompt_tokens: u.prompt_token_count,
            completion_tokens: u.candidates_token_count,
            total_tokens: u.total_token_count,
        }).unwrap_or(OpenAIUsage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
        });

        OpenAIChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: model.to_string(),
            choices: vec![choice],
            usage,
        }
    }

    /// Get the endpoint URL for the model
    fn get_endpoint_url(&self, model: &str, stream: bool) -> String {
        let method = if stream {
            "streamGenerateContent"
        } else {
            "generateContent"
        };

        format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:{}",
            self.location, self.project_id, self.location, model, method
        )
    }

    /// Classify Vertex AI errors
    fn classify_error(error_detail: &str) -> String {
        let error_lower = error_detail.to_lowercase();

        if error_lower.contains("unauthorized") || error_lower.contains("authentication") {
            return "Invalid access token. Please check your Google Cloud authentication."
                .to_string();
        }

        if error_lower.contains("quota") || error_lower.contains("rate") {
            return "Rate limit or quota exceeded. Please check your Google Cloud quota."
                .to_string();
        }

        if error_lower.contains("not found") || error_lower.contains("model") {
            return "Model not found or not available in your region.".to_string();
        }

        if error_lower.contains("permission") {
            return "Permission denied. Please check your Google Cloud IAM permissions.".to_string();
        }

        error_detail.to_string()
    }

    /// Internal method to send completion request
    async fn send_completion_request(
        &self,
        request: &OpenAIChatCompletionRequest,
        _cancel_notify: Option<Arc<Notify>>,
    ) -> Result<OpenAIChatCompletionResponse, ProviderError> {
        let url = self.get_endpoint_url(&request.model, false);
        let vertex_request = self.convert_request_to_vertex(request);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .bearer_auth(&self.access_token)
            .json(&vertex_request)
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
                401 | 403 => ProviderError::Authentication(classified_error),
                429 => ProviderError::RateLimit(classified_error),
                400 | 404 => ProviderError::BadRequest(classified_error),
                _ => ProviderError::ApiError {
                    status: status.as_u16(),
                    message: classified_error,
                },
            });
        }

        let vertex_response: VertexAIResponse = response
            .json()
            .await
            .map_err(|e| ProviderError::Unexpected(format!("Failed to parse response: {}", e)))?;

        Ok(self.convert_response_from_vertex(vertex_response, &request.model))
    }

    /// Internal method to send streaming request
    async fn send_stream_request(
        &self,
        request: &OpenAIChatCompletionRequest,
    ) -> Result<reqwest::Response, ProviderError> {
        let url = self.get_endpoint_url(&request.model, true);
        let vertex_request = self.convert_request_to_vertex(request);

        let response = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .bearer_auth(&self.access_token)
            .json(&vertex_request)
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
                401 | 403 => ProviderError::Authentication(classified_error),
                429 => ProviderError::RateLimit(classified_error),
                400 | 404 => ProviderError::BadRequest(classified_error),
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
impl Provider for VertexAIProvider {
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
        request: OpenAIChatCompletionRequest,
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
        "Vertex AI"
    }
}
