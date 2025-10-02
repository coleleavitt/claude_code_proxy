//! API endpoint handlers
//!
//! This module implements the HTTP endpoints for the Claude-to-OpenAI proxy,
//! including message creation, token counting, and health checks.

use crate::conversion::request_converter::convert_claude_to_openai;
use crate::conversion::response_converter::{
    convert_openai_streaming_to_claude_with_cancellation, convert_openai_to_claude,
};
use crate::core::config::Config;
use crate::core::model_manager::ModelManager;
use crate::core::provider::Provider;
use crate::models::claude::{ClaudeMessagesRequest, ClaudeTokenCountRequest, MessageContent};
use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response, Sse},
    routing::{get, post},
};
use futures::StreamExt;
use serde_json::json;
use std::convert::Infallible;
use std::sync::Arc;
use tracing::{debug, error, warn};

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub model_manager: Arc<ModelManager>,
    pub provider: Arc<dyn Provider>,
}

/// Create the API router with all endpoints
pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(root))
        .route("/v1/messages", post(create_message))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .route("/health", get(health_check))
        .route("/test-connection", get(test_connection))
        .with_state(state)
}

/// Validate API key from request headers
fn validate_api_key(headers: &HeaderMap, config: &Config) -> Result<(), StatusCode> {
    // Extract API key from headers
    let client_api_key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .or_else(|| {
            headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "))
        });

    // Skip validation if ANTHROPIC_API_KEY is not set
    if config.anthropic_api_key.is_none() {
        return Ok(());
    }

    // Validate the client API key
    match client_api_key {
        Some(key) if config.validate_client_api_key(key) => Ok(()),
        _ => {
            warn!("Invalid API key provided by client");
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

/// POST /v1/messages - Create a message
async fn create_message(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ClaudeMessagesRequest>,
) -> Result<Response, StatusCode> {
    // Validate API key
    validate_api_key(&headers, &state.config)?;

    // Capture request details before move
    let model_name = request.model.clone();
    let stream = request.stream;

    // Log incoming request details
    tracing::info!(
        "📥 Incoming Claude API request: model={}, stream={}, messages={}",
        model_name,
        stream,
        request.messages.len()
    );

    // Log the full request payload in debug mode
    debug!("Full request payload: {:?}", request);

    debug!(
        "Processing Claude request: model={}, stream={}",
        model_name, stream
    );

    // Generate unique request ID for cancellation tracking
    let request_id = uuid::Uuid::new_v4().to_string();

    // Apply context truncation if needed
    let messages = if request.messages.len() > state.config.max_messages_limit as usize {
        let original_count = request.messages.len();
        let truncated_messages: Vec<crate::models::claude::ClaudeMessage> = request
            .messages
            .iter()
            .skip(original_count - state.config.max_messages_limit as usize)
            .cloned()
            .collect();

        tracing::warn!(
            "📜 Context truncated: {} messages → {} messages (removed {} oldest messages)",
            original_count,
            truncated_messages.len(),
            original_count - truncated_messages.len()
        );
        truncated_messages
    } else {
        request.messages.clone()
    };

    // Create a processed request for conversion
    // Move the request here to avoid multiple borrows
    let mut processed_request = request;
    processed_request.messages = messages;
    processed_request.model = model_name.clone();
    processed_request.stream = stream;

    // Convert Claude request to OpenAI format
    let openai_request = convert_claude_to_openai(
        &processed_request,
        &state.model_manager,
        state.config.min_tokens_limit,
        state.config.max_tokens_limit,
    );

    if stream {
        // Streaming response with client disconnection detection
        match state
            .provider
            .create_chat_completion_stream(openai_request, Some(request_id.clone()))
            .await
        {
            Ok(provider_stream) => {
                // Wrap ProviderError in a String-based error for the stream
                #[derive(Debug)]
                struct StreamError(String);
                impl std::fmt::Display for StreamError {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        write!(f, "{}", self.0)
                    }
                }
                impl std::error::Error for StreamError {}

                // Convert provider stream to Claude SSE format with cancellation support
                let provider = state.provider.clone();
                let model_name = model_name.clone();

                let claude_stream = convert_openai_streaming_to_claude_with_cancellation(
                    provider_stream.map(|r| r.map_err(|e| StreamError(e.to_string()))),
                    model_name,
                    provider,
                    request_id,
                )
                .await;

                // Convert Result<String, String> to SSE events
                let sse_stream = claude_stream.map(|item| match item {
                    Ok(data) => {
                        Ok::<_, Infallible>(axum::response::sse::Event::default().data(data))
                    }
                    Err(e) => {
                        error!("Stream error: {}", e);
                        Ok(axum::response::sse::Event::default()
                            .data(format!("event: error\ndata: {}\n\n", e)))
                    }
                });

                // Create SSE response with proper headers
                let mut response = Sse::new(sse_stream)
                    .keep_alive(axum::response::sse::KeepAlive::default())
                    .into_response();

                // Add CORS and cache control headers
                let response_headers = response.headers_mut();
                response_headers.insert("Cache-Control", "no-cache".parse().unwrap());
                response_headers.insert("Connection", "keep-alive".parse().unwrap());
                Ok(response)
            }
            Err(e) => {
                error!("Provider streaming error: {}", e);
                let error_response = json!({
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": e.to_string()
                    }
                });
                Ok((StatusCode::INTERNAL_SERVER_ERROR, Json(error_response)).into_response())
            }
        }
    } else {
        // Non-streaming response
        match state
            .provider
            .create_chat_completion(&openai_request, None)
            .await
        {
            Ok(provider_response) => {
                let claude_response = convert_openai_to_claude(&provider_response, &model_name);
                Ok(Json(claude_response).into_response())
            }
            Err(e) => {
                error!("Provider API error: {}", e);
                let error_response = json!({
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": e.to_string()
                    }
                });
                Ok((StatusCode::INTERNAL_SERVER_ERROR, Json(error_response)).into_response())
            }
        }
    }
}

/// POST /v1/messages/count_tokens - Count tokens in a request
async fn count_tokens(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ClaudeTokenCountRequest>,
) -> Result<Response, StatusCode> {
    // Validate API key
    validate_api_key(&headers, &state.config)?;

    debug!("Token counting for model: {}", request.model);

    // Simple character-based token estimation
    // Rough estimation: 4 characters per token
    let mut total_chars = 0;

    // Count system message characters
    if let Some(ref system) = request.system {
        use crate::models::claude::SystemContent;
        match system {
            SystemContent::String(s) => total_chars += s.len(),
            SystemContent::Blocks(blocks) => {
                for block in blocks {
                    total_chars += block.text.len();
                }
            }
        }
    }

    // Count message characters
    for msg in &request.messages {
        match &msg.content {
            MessageContent::String(s) => total_chars += s.len(),
            MessageContent::Blocks(blocks) => {
                for block in blocks {
                    use crate::models::claude::ClaudeContentBlock;
                    if let ClaudeContentBlock::Text(text_block) = block {
                        total_chars += text_block.text.len();
                    }
                }
            }
        }
    }

    // Rough estimation: 4 characters per token
    let estimated_tokens = std::cmp::max(1, total_chars / 4);

    let response = json!({
        "input_tokens": estimated_tokens
    });

    Ok(Json(response).into_response())
}

/// GET / - Root endpoint
async fn root(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!({
        "message": "Claude-to-OpenAI API Proxy v1.0.0",
        "status": "running",
        "config": {
            "openai_base_url": state.config.openai_base_url,
            "max_tokens_limit": state.config.max_tokens_limit,
            "api_key_configured": !state.config.openai_api_key.is_empty(),
            "client_api_key_validation": state.config.anthropic_api_key.is_some(),
            "big_model": state.config.big_model,
            "middle_model": state.config.middle_model,
            "small_model": state.config.small_model,
        },
        "endpoints": {
            "messages": "/v1/messages",
            "count_tokens": "/v1/messages/count_tokens",
            "health": "/health",
            "test_connection": "/test-connection",
        },
    }))
}

/// GET /health - Health check endpoint
async fn health_check(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "openai_api_configured": !state.config.openai_api_key.is_empty(),
        "api_key_valid": state.config.validate_api_key(),
        "client_api_key_validation": state.config.anthropic_api_key.is_some(),
    }))
}

/// GET /test-connection - Test OpenAI API connectivity
async fn test_connection(State(state): State<AppState>) -> impl IntoResponse {
    // Actual API test
    use crate::models::openai::{OpenAIChatCompletionRequest, OpenAIMessage};

    let test_request = OpenAIChatCompletionRequest {
        model: state.config.small_model.clone(),
        messages: vec![OpenAIMessage {
            role: "user".to_string(),
            content: Some(serde_json::Value::String("Hello".to_string())),
            tool_calls: None,
            tool_call_id: None,
        }],
        max_tokens: Some(5),
        temperature: Some(1.0),
        top_p: None,
        stop: None,
        stream: false,
        stream_options: None,
        tools: None,
        tool_choice: None,
    };

    match state
        .provider
        .create_chat_completion(&test_request, None)
        .await
    {
        Ok(response) => Json(json!({
            "status": "success",
            "message": format!("Successfully connected to {} API", state.provider.provider_name()),
            "provider": state.provider.provider_name(),
            "model_used": state.config.small_model,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "response_id": response.id,
        })),
        Err(e) => {
            error!("API connectivity test failed: {}", e);
            Json(json!({
                "status": "failed",
                "error_type": "API Error",
                "message": e.to_string(),
                "provider": state.provider.provider_name(),
                "timestamp": chrono::Utc::now().to_rfc3339(),
                "suggestions": [
                    "Check your API key is valid",
                    "Verify your API key has the necessary permissions",
                    "Check if you have reached rate limits",
                ],
            }))
        }
    }
}
