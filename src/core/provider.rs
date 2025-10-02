//! Provider abstraction layer for different LLM API providers
//!
//! This module defines a common trait for different providers (OpenAI, OpenRouter, Vertex AI)
//! and provides factory methods for creating provider instances.

use crate::models::openai::{OpenAIChatCompletionRequest, OpenAIChatCompletionResponse};
use async_trait::async_trait;
use futures::stream::Stream;
use std::pin::Pin;
use thiserror::Error;

/// Error types for provider operations
#[derive(Debug, Error)]
pub enum ProviderError {
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

/// Trait for LLM API providers
#[async_trait]
pub trait Provider: Send + Sync {
    /// Send non-streaming chat completion request
    async fn create_chat_completion(
        &self,
        request: &OpenAIChatCompletionRequest,
        request_id: Option<String>,
    ) -> Result<OpenAIChatCompletionResponse, ProviderError>;

    /// Send streaming chat completion request
    async fn create_chat_completion_stream(
        &self,
        request: OpenAIChatCompletionRequest,
        request_id: Option<String>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, ProviderError>> + Send>>, ProviderError>;

    /// Cancel an active request by request_id
    async fn cancel_request(&self, request_id: &str) -> bool;

    /// Get the provider name
    fn provider_name(&self) -> &str;
}

/// Supported provider types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderType {
    OpenAI,
    OpenRouter,
    VertexAI,
}

impl ProviderType {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "openai" => Some(ProviderType::OpenAI),
            "openrouter" => Some(ProviderType::OpenRouter),
            "vertexai" | "vertex-ai" | "vertex_ai" => Some(ProviderType::VertexAI),
            _ => None,
        }
    }
}