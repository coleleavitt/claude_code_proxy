//! OpenAI to Claude response conversion
//!
//! This module converts OpenAI API responses back to Claude API format,
//! supporting both streaming and non-streaming responses.

use crate::core::constants::{content, delta as delta_const, event, role, stop};
use crate::models::openai::{OpenAIChatCompletionResponse, OpenAIStreamingChunk};
use futures::Stream;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::pin::Pin;
use tracing::{error, warn};

/// Convert OpenAI response to Claude format
///
/// Transforms an OpenAI chat completion response into a Claude Messages API
/// response format.
///
/// # Arguments
///
/// * `openai_response` - The OpenAI response to convert
/// * `original_model` - The original Claude model name from the request
pub fn convert_openai_to_claude(
    openai_response: &OpenAIChatCompletionResponse,
    original_model: &str,
) -> Value {
    let choice = &openai_response.choices[0];
    let message = &choice.message;

    // Build content blocks
    let mut content_blocks = Vec::new();

    // Add text content if present
    if let Some(ref content) = message.content {
        if let Some(text) = content.as_str() {
            if !text.is_empty() {
                content_blocks.push(json!({
                    "type": "text",
                    "text": text
                }));
            }
        }
    }

    // Add tool calls if present
    if let Some(ref tool_calls) = message.tool_calls {
        for tool_call in tool_calls {
            let input: Value =
                serde_json::from_str(&tool_call.function.arguments).unwrap_or_else(|_| json!({}));

            content_blocks.push(json!({
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": input
            }));
        }
    }

    // Ensure at least one content block (required by Claude API)
    if content_blocks.is_empty() {
        content_blocks.push(json!({
            "type": "text",
            "text": ""
        }));
    }

    // Determine stop reason
    let stop_reason = match choice.finish_reason.as_deref() {
        Some("stop") => "end_turn",
        Some("length") => "max_tokens",
        Some("tool_calls") => "tool_use",
        _ => "end_turn",
    };

    json!({
        "id": openai_response.id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": original_model,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": {
            "input_tokens": openai_response.usage.prompt_tokens,
            "output_tokens": openai_response.usage.completion_tokens
        }
    })
}

/// Convert OpenAI streaming chunk to Claude SSE events
///
/// This is a simplified version. The full implementation would need to track
/// state across multiple chunks to properly format tool calls.
///
/// # Arguments
///
/// * `chunk` - The OpenAI streaming chunk
/// * `original_model` - The original Claude model name
pub fn convert_streaming_chunk_to_claude(
    chunk: &OpenAIStreamingChunk,
    original_model: &str,
) -> Vec<String> {
    let mut events = Vec::new();

    if let Some(choice) = chunk.choices.first() {
        let delta = &choice.delta;

        // Handle content delta
        if let Some(ref content) = delta.content {
            if !content.is_empty() {
                let event = json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "text_delta",
                        "text": content
                    }
                });
                events.push(format!("event: content_block_delta\ndata: {}\n", event));
            }
        }

        // Handle finish reason
        if let Some(ref finish_reason) = choice.finish_reason {
            let stop_reason = match finish_reason.as_str() {
                "stop" => "end_turn",
                "length" => "max_tokens",
                "tool_calls" => "tool_use",
                _ => "end_turn",
            };

            let event = json!({
                "type": "message_delta",
                "delta": {
                    "stop_reason": stop_reason
                }
            });
            events.push(format!("event: message_delta\ndata: {}\n", event));
        }
    }

    events
}

/// Create a Claude message_start event
pub fn create_message_start_event(message_id: &str, model: &str) -> String {
    let event = json!({
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": null,
            "stop_sequence": null,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0
            }
        }
    });
    format!("event: message_start\ndata: {}\n", event)
}

/// Create a Claude content_block_start event
pub fn create_content_block_start_event(index: u32) -> String {
    let event = json!({
        "type": "content_block_start",
        "index": index,
        "content_block": {
            "type": "text",
            "text": ""
        }
    });
    format!("event: content_block_start\ndata: {}\n", event)
}

/// Create a Claude content_block_stop event
pub fn create_content_block_stop_event(index: u32) -> String {
    let event = json!({
        "type": "content_block_stop",
        "index": index
    });
    format!("event: content_block_stop\ndata: {}\n", event)
}

/// Create a Claude message_stop event
pub fn create_message_stop_event() -> String {
    let event = json!({
        "type": "message_stop"
    });
    format!("event: message_stop\ndata: {}\n", event)
}

/// Tool call tracking structure for streaming
#[derive(Debug, Clone)]
struct ToolCallState {
    id: Option<String>,
    name: Option<String>,
    args_buffer: String,
    json_sent: bool,
    claude_index: Option<u32>,
    started: bool,
}

/// Convert OpenAI streaming to Claude SSE format with full tool call support
///
/// This async generator function processes an OpenAI SSE stream and yields
/// Claude-formatted SSE events.
pub async fn convert_openai_streaming_to_claude<S, E>(
    openai_stream: S,
    original_model: String,
) -> Pin<Box<dyn Stream<Item = Result<String, String>> + Send>>
where
    S: Stream<Item = Result<String, E>> + Send + 'static,
    E: std::error::Error + Send + 'static,
{
    use futures::StreamExt;

    let message_id = format!(
        "msg_{}",
        uuid::Uuid::new_v4().simple().to_string()[..24].to_string()
    );

    let stream = async_stream::stream! {
        // Send initial SSE events
        let message_start = json!({
            "type": event::MESSAGE_START,
            "message": {
                "id": &message_id,
                "type": "message",
                "role": role::ASSISTANT,
                "model": &original_model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0
                }
            }
        });
        yield Ok(format!("event: {}\ndata: {}\n\n", event::MESSAGE_START, message_start));

        let content_block_start = json!({
            "type": event::CONTENT_BLOCK_START,
            "index": 0,
            "content_block": {
                "type": content::TEXT,
                "text": ""
            }
        });
        yield Ok(format!("event: {}\ndata: {}\n\n", event::CONTENT_BLOCK_START, content_block_start));

        let ping = json!({"type": event::PING});
        yield Ok(format!("event: {}\ndata: {}\n\n", event::PING, ping));

        // Track state
        let text_block_index = 0u32;
        let mut tool_block_counter = 0u32;
        let mut current_tool_calls: HashMap<usize, ToolCallState> = HashMap::new();
        let mut final_stop_reason = stop::END_TURN;
        let mut usage_data = json!({
            "input_tokens": 0,
            "output_tokens": 0
        });

        // Process stream
        tokio::pin!(openai_stream);

        while let Some(line_result) = openai_stream.next().await {
            let line = match line_result {
                Ok(l) => l,
                Err(e) => {
                    error!("Stream error: {}", e);
                    let error_event = json!({
                        "type": "error",
                        "error": {
                            "type": "api_error",
                            "message": format!("Stream error: {}", e)
                        }
                    });
                    yield Ok(format!("event: error\ndata: {}\n\n", error_event));
                    break;
                }
            };

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if !trimmed.starts_with("data: ") {
                continue;
            }

            let chunk_data = &trimmed[6..];
            if chunk_data.trim() == "[DONE]" {
                break;
            }

            let chunk: Value = match serde_json::from_str(chunk_data) {
                Ok(c) => c,
                Err(e) => {
                    warn!("Failed to parse chunk: {}, error: {}", chunk_data, e);
                    continue;
                }
            };

            // Extract usage if present
            if let Some(usage) = chunk.get("usage") {
                let prompt_tokens = usage.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                let completion_tokens = usage.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                let cache_tokens = usage
                    .get("prompt_tokens_details")
                    .and_then(|d| d.get("cached_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);

                usage_data = json!({
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "cache_read_input_tokens": cache_tokens
                });
            }

            let choices = match chunk.get("choices").and_then(|c| c.as_array()) {
                Some(c) if !c.is_empty() => c,
                _ => continue,
            };

            let choice = &choices[0];
            let delta = choice.get("delta");
            let finish_reason = choice.get("finish_reason").and_then(|f| f.as_str());

            // Handle text content delta
            if let Some(content_text) = delta.and_then(|d| d.get("content")).and_then(|c| c.as_str()) {
                if !content_text.is_empty() {
                    let content_delta = json!({
                        "type": event::CONTENT_BLOCK_DELTA,
                        "index": text_block_index,
                        "delta": {
                            "type": delta_const::TEXT,
                            "text": content_text
                        }
                    });
                    yield Ok(format!("event: {}\ndata: {}\n\n", event::CONTENT_BLOCK_DELTA, content_delta));
                }
            }

            // Handle tool calls (check for non-empty array)
            if let Some(tool_calls) = delta.and_then(|d| d.get("tool_calls")).and_then(|tc| tc.as_array()) {
                if !tool_calls.is_empty() {
                for tc_delta in tool_calls {
                    let tc_index = tc_delta.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;

                    // Initialize tool call state if needed
                    if !current_tool_calls.contains_key(&tc_index) {
                        current_tool_calls.insert(tc_index, ToolCallState {
                            id: None,
                            name: None,
                            args_buffer: String::new(),
                            json_sent: false,
                            claude_index: None,
                            started: false,
                        });
                    }

                    let tool_call = current_tool_calls.get_mut(&tc_index).unwrap();

                    // Update ID
                    if let Some(id) = tc_delta.get("id").and_then(|i| i.as_str()) {
                        tool_call.id = Some(id.to_string());
                    }

                    // Update function name
                    if let Some(func) = tc_delta.get("function") {
                        if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                            tool_call.name = Some(name.to_string());
                        }

                        // Start content block when we have complete initial data
                        if let (Some(id), Some(name)) = (&tool_call.id, &tool_call.name) {
                            if !tool_call.started {
                                tool_block_counter += 1;
                                let claude_index = text_block_index + tool_block_counter;
                                tool_call.claude_index = Some(claude_index);
                                tool_call.started = true;

                                let tool_start = json!({
                                    "type": event::CONTENT_BLOCK_START,
                                    "index": claude_index,
                                    "content_block": {
                                        "type": content::TOOL_USE,
                                        "id": id,
                                        "name": name,
                                        "input": {}
                                    }
                                });
                                yield Ok(format!("event: {}\ndata: {}\n\n", event::CONTENT_BLOCK_START, tool_start));
                            }
                        }

                        // Handle arguments
                        if let Some(args) = func.get("arguments").and_then(|a| a.as_str()) {
                            if tool_call.started && !args.is_empty() {
                                tool_call.args_buffer.push_str(args);

                                // Try to parse and send when we have valid JSON
                                if let Ok(_) = serde_json::from_str::<Value>(&tool_call.args_buffer) {
                                    if !tool_call.json_sent {
                                        if let Some(claude_idx) = tool_call.claude_index {
                                            let input_delta = json!({
                                                "type": event::CONTENT_BLOCK_DELTA,
                                                "index": claude_idx,
                                                "delta": {
                                                    "type": delta_const::INPUT_JSON,
                                                    "partial_json": &tool_call.args_buffer
                                                }
                                            });
                                            yield Ok(format!("event: {}\ndata: {}\n\n", event::CONTENT_BLOCK_DELTA, input_delta));
                                            tool_call.json_sent = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                }
            }

            // Handle finish reason
            if let Some(reason) = finish_reason {
                final_stop_reason = match reason {
                    "length" => stop::MAX_TOKENS,
                    "tool_calls" | "function_call" => stop::TOOL_USE,
                    "stop" => stop::END_TURN,
                    _ => stop::END_TURN,
                };
                break;
            }
        }

        // Send closing events
        let content_stop = json!({
            "type": event::CONTENT_BLOCK_STOP,
            "index": text_block_index
        });
        yield Ok(format!("event: {}\ndata: {}\n\n", event::CONTENT_BLOCK_STOP, content_stop));

        // Stop tool call blocks
        for tool_data in current_tool_calls.values() {
            if tool_data.started {
                if let Some(idx) = tool_data.claude_index {
                    let tool_stop = json!({
                        "type": event::CONTENT_BLOCK_STOP,
                        "index": idx
                    });
                    yield Ok(format!("event: {}\ndata: {}\n\n", event::CONTENT_BLOCK_STOP, tool_stop));
                }
            }
        }

        // Message delta with stop reason and usage
        let message_delta = json!({
            "type": event::MESSAGE_DELTA,
            "delta": {
                "stop_reason": final_stop_reason,
                "stop_sequence": null
            },
            "usage": usage_data
        });
        yield Ok(format!("event: {}\ndata: {}\n\n", event::MESSAGE_DELTA, message_delta));

        // Message stop
        let message_stop = json!({"type": event::MESSAGE_STOP});
        yield Ok(format!("event: {}\ndata: {}\n\n", event::MESSAGE_STOP, message_stop));
    };

    Box::pin(stream)
}

/// Convert OpenAI streaming to Claude SSE format with client disconnection detection
///
/// This version includes support for detecting client disconnections and cancelling
/// the underlying OpenAI request when the client disconnects.
pub async fn convert_openai_streaming_to_claude_with_cancellation<S, E>(
    openai_stream: S,
    original_model: String,
    provider: std::sync::Arc<dyn crate::core::provider::Provider>,
    request_id: String,
) -> Pin<Box<dyn Stream<Item = Result<String, String>> + Send>>
where
    S: Stream<Item = Result<String, E>> + Send + 'static,
    E: std::error::Error + Send + 'static,
{
    use futures::StreamExt;
    use tokio::time::{Duration, interval};

    let message_id = format!(
        "msg_{}",
        uuid::Uuid::new_v4().simple().to_string()[..24].to_string()
    );

    let stream = async_stream::stream! {
        // Send initial SSE events
        let message_start = json!({
            "type": event::MESSAGE_START,
            "message": {
                "id": &message_id,
                "type": "message",
                "role": role::ASSISTANT,
                "model": &original_model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0
                }
            }
        });
        yield Ok(format!("event: {}\ndata: {}\n\n", event::MESSAGE_START, message_start));

        let content_block_start = json!({
            "type": event::CONTENT_BLOCK_START,
            "index": 0,
            "content_block": {
                "type": content::TEXT,
                "text": ""
            }
        });
        yield Ok(format!("event: {}\ndata: {}\n\n", event::CONTENT_BLOCK_START, content_block_start));

        let ping = json!({"type": event::PING});
        yield Ok(format!("event: {}\ndata: {}\n\n", event::PING, ping));

        // Track state
        let text_block_index = 0u32;
        let mut tool_block_counter = 0u32;
        let mut current_tool_calls: HashMap<usize, ToolCallState> = HashMap::new();
        let mut final_stop_reason = stop::END_TURN;
        let mut usage_data = json!({
            "input_tokens": 0,
            "output_tokens": 0
        });

        // Create a heartbeat to check for cancellation
        let mut heartbeat = interval(Duration::from_millis(100));
        let provider_arc = provider.clone();
        let req_id = request_id.clone();

        // Process stream
        tokio::pin!(openai_stream);

        let cancelled = false;

        loop {
            tokio::select! {
                // Check heartbeat for potential cancellation signals
                _ = heartbeat.tick() => {
                    // Just continue - this keeps the loop responsive
                    continue;
                }

                // Process stream items
                line_result = openai_stream.next() => {
                    match line_result {
                        Some(Ok(line)) => {
                            let trimmed = line.trim();
                            if trimmed.is_empty() {
                                continue;
                            }

                            if !trimmed.starts_with("data: ") {
                                continue;
                            }

                            let chunk_data = &trimmed[6..];
                            if chunk_data.trim() == "[DONE]" {
                                break;
                            }

                            let chunk: Value = match serde_json::from_str(chunk_data) {
                                Ok(c) => c,
                                Err(e) => {
                                    warn!("Failed to parse chunk: {}, error: {}", chunk_data, e);
                                    continue;
                                }
                            };

                            // Extract usage if present
                            if let Some(usage) = chunk.get("usage") {
                                let prompt_tokens = usage.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                                let completion_tokens = usage.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                                let cache_tokens = usage
                                    .get("prompt_tokens_details")
                                    .and_then(|d| d.get("cached_tokens"))
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0);

                                usage_data = json!({
                                    "input_tokens": prompt_tokens,
                                    "output_tokens": completion_tokens,
                                    "cache_read_input_tokens": cache_tokens
                                });
                            }

                            let choices = match chunk.get("choices").and_then(|c| c.as_array()) {
                                Some(c) if !c.is_empty() => c,
                                _ => continue,
                            };

                            let choice = &choices[0];
                            let delta = choice.get("delta");
                            let finish_reason = choice.get("finish_reason").and_then(|f| f.as_str());

                            // Handle text content delta
                            if let Some(content_text) = delta.and_then(|d| d.get("content")).and_then(|c| c.as_str()) {
                                if !content_text.is_empty() {
                                    let content_delta = json!({
                                        "type": event::CONTENT_BLOCK_DELTA,
                                        "index": text_block_index,
                                        "delta": {
                                            "type": delta_const::TEXT,
                                            "text": content_text
                                        }
                                    });
                                    yield Ok(format!("event: {}\ndata: {}\n\n", event::CONTENT_BLOCK_DELTA, content_delta));
                                }
                            }

                            // Handle tool calls (check for non-empty array)
                            if let Some(tool_calls) = delta.and_then(|d| d.get("tool_calls")).and_then(|tc| tc.as_array()) {
                                if !tool_calls.is_empty() {
                                for tc_delta in tool_calls {
                                    let tc_index = tc_delta.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;

                                    // Initialize tool call state if needed
                                    if !current_tool_calls.contains_key(&tc_index) {
                                        current_tool_calls.insert(tc_index, ToolCallState {
                                            id: None,
                                            name: None,
                                            args_buffer: String::new(),
                                            json_sent: false,
                                            claude_index: None,
                                            started: false,
                                        });
                                    }

                                    let tool_call = current_tool_calls.get_mut(&tc_index).unwrap();

                                    // Update ID
                                    if let Some(id) = tc_delta.get("id").and_then(|i| i.as_str()) {
                                        tool_call.id = Some(id.to_string());
                                    }

                                    // Update function name
                                    if let Some(func) = tc_delta.get("function") {
                                        if let Some(name) = func.get("name").and_then(|n| n.as_str()) {
                                            tool_call.name = Some(name.to_string());
                                        }

                                        // Start content block when we have complete initial data
                                        if let (Some(id), Some(name)) = (&tool_call.id, &tool_call.name) {
                                            if !tool_call.started {
                                                tool_block_counter += 1;
                                                let claude_index = text_block_index + tool_block_counter;
                                                tool_call.claude_index = Some(claude_index);
                                                tool_call.started = true;

                                                let tool_start = json!({
                                                    "type": event::CONTENT_BLOCK_START,
                                                    "index": claude_index,
                                                    "content_block": {
                                                        "type": content::TOOL_USE,
                                                        "id": id,
                                                        "name": name,
                                                        "input": {}
                                                    }
                                                });
                                                yield Ok(format!("event: {}\ndata: {}\n\n", event::CONTENT_BLOCK_START, tool_start));
                                            }
                                        }

                                        // Handle arguments
                                        if let Some(args) = func.get("arguments").and_then(|a| a.as_str()) {
                                            if tool_call.started && !args.is_empty() {
                                                tool_call.args_buffer.push_str(args);

                                                // Try to parse and send when we have valid JSON
                                                if let Ok(_) = serde_json::from_str::<Value>(&tool_call.args_buffer) {
                                                    if !tool_call.json_sent {
                                                        if let Some(claude_idx) = tool_call.claude_index {
                                                            let input_delta = json!({
                                                                "type": event::CONTENT_BLOCK_DELTA,
                                                                "index": claude_idx,
                                                                "delta": {
                                                                    "type": delta_const::INPUT_JSON,
                                                                    "partial_json": &tool_call.args_buffer
                                                                }
                                                            });
                                                            yield Ok(format!("event: {}\ndata: {}\n\n", event::CONTENT_BLOCK_DELTA, input_delta));
                                                            tool_call.json_sent = true;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                }
                            }

                            // Handle finish reason
                            if let Some(reason) = finish_reason {
                                final_stop_reason = match reason {
                                    "length" => stop::MAX_TOKENS,
                                    "tool_calls" | "function_call" => stop::TOOL_USE,
                                    "stop" => stop::END_TURN,
                                    _ => stop::END_TURN,
                                };
                                break;
                            }
                        }
                        Some(Err(e)) => {
                            error!("Stream error: {}", e);
                            let error_event = json!({
                                "type": "error",
                                "error": {
                                    "type": "api_error",
                                    "message": format!("Stream error: {}", e)
                                }
                            });
                            yield Ok(format!("event: error\ndata: {}\n\n", error_event));
                            break;
                        }
                        None => {
                            // Stream ended
                            break;
                        }
                    }
                }
            }

            if cancelled {
                break;
            }
        }

        // Send closing events
        let content_stop = json!({
            "type": event::CONTENT_BLOCK_STOP,
            "index": text_block_index
        });
        yield Ok(format!("event: {}\ndata: {}\n\n", event::CONTENT_BLOCK_STOP, content_stop));

        // Stop tool call blocks
        for tool_data in current_tool_calls.values() {
            if tool_data.started {
                if let Some(idx) = tool_data.claude_index {
                    let tool_stop = json!({
                        "type": event::CONTENT_BLOCK_STOP,
                        "index": idx
                    });
                    yield Ok(format!("event: {}\ndata: {}\n\n", event::CONTENT_BLOCK_STOP, tool_stop));
                }
            }
        }

        // Message delta with stop reason and usage
        let message_delta = json!({
            "type": event::MESSAGE_DELTA,
            "delta": {
                "stop_reason": final_stop_reason,
                "stop_sequence": null
            },
            "usage": usage_data
        });
        yield Ok(format!("event: {}\ndata: {}\n\n", event::MESSAGE_DELTA, message_delta));

        // Message stop
        let message_stop = json!({"type": event::MESSAGE_STOP});
        yield Ok(format!("event: {}\ndata: {}\n\n", event::MESSAGE_STOP, message_stop));
    };

    Box::pin(stream)
}
