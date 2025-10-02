//! Claude to OpenAI request conversion
//!
//! This module converts Claude API request format to OpenAI API format,
//! handling message transformation, tool conversion, and parameter mapping.

use crate::core::constants::{content, role, tool};
use crate::core::model_manager::ModelManager;
use crate::models::claude::{
    ClaudeContentBlock, ClaudeMessage, ClaudeMessagesRequest, MessageContent, SystemContent,
    ToolResultContent,
};
use crate::models::openai::{
    OpenAIChatCompletionRequest, OpenAIFunctionDef, OpenAIMessage, OpenAITool, OpenAIToolCall,
};
use serde_json::Value;
use std::collections::HashMap;
use tracing::debug;

/// Convert Claude API request to OpenAI format
///
/// Transforms a Claude Messages API request into an equivalent OpenAI
/// chat completion request, including message conversion, tool mapping,
/// and parameter translation.
///
/// # Arguments
///
/// * `claude_request` - The Claude API request to convert
/// * `model_manager` - Model manager for mapping Claude models to OpenAI models
/// * `min_tokens` - Minimum token limit
/// * `max_tokens` - Maximum token limit
pub fn convert_claude_to_openai(
    claude_request: &ClaudeMessagesRequest,
    model_manager: &ModelManager,
    min_tokens: u32,
    max_tokens: u32,
) -> OpenAIChatCompletionRequest {
    // Map model
    let openai_model = model_manager.map_claude_model_to_openai(&claude_request.model);

    // Convert messages
    let mut openai_messages = Vec::new();

    // Add system message if present
    if let Some(ref system) = claude_request.system {
        let system_text = match system {
            SystemContent::String(s) => s.clone(),
            SystemContent::Blocks(blocks) => {
                let text_parts: Vec<String> = blocks
                    .iter()
                    .filter(|block| block.content_type == content::TEXT)
                    .map(|block| block.text.clone())
                    .collect();
                text_parts.join("\n\n")
            }
        };

        if !system_text.trim().is_empty() {
            openai_messages.push(OpenAIMessage {
                role: role::SYSTEM.to_string(),
                content: Some(Value::String(system_text.trim().to_string())),
                tool_calls: None,
                tool_call_id: None,
            });
        }
    }

    // Process Claude messages
    let mut i = 0;
    while i < claude_request.messages.len() {
        let msg = &claude_request.messages[i];

        if msg.role == role::USER {
            let openai_message = convert_claude_user_message(msg);
            openai_messages.push(openai_message);
        } else if msg.role == role::ASSISTANT {
            let openai_message = convert_claude_assistant_message(msg);
            openai_messages.push(openai_message);

            // Check if next message contains tool results
            if i + 1 < claude_request.messages.len() {
                let next_msg = &claude_request.messages[i + 1];
                if next_msg.role == role::USER && has_tool_results(next_msg) {
                    // Process tool results
                    i += 1; // Skip to tool result message
                    let tool_results = convert_claude_tool_results(next_msg);
                    openai_messages.extend(tool_results);
                }
            }
        }

        i += 1;
    }

    // Clamp max_tokens to configured limits
    let clamped_max_tokens = claude_request
        .max_tokens
        .max(min_tokens)
        .min(max_tokens);

    // Build OpenAI request
    let mut openai_request = OpenAIChatCompletionRequest {
        model: openai_model,
        messages: openai_messages,
        max_tokens: Some(clamped_max_tokens),
        temperature: Some(claude_request.temperature),
        top_p: claude_request.top_p,
        stop: claude_request.stop_sequences.clone(),
        stream: claude_request.stream,
        stream_options: None, // Will be set by client if streaming
        tools: None,
        tool_choice: None,
    };

    // Convert tools
    if let Some(ref claude_tools) = claude_request.tools {
        let openai_tools: Vec<OpenAITool> = claude_tools
            .iter()
            .filter(|tool| !tool.name.trim().is_empty())
            .map(|tool| OpenAITool {
                tool_type: tool::FUNCTION.to_string(),
                function: OpenAIFunctionDef {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: tool.input_schema.clone(),
                },
            })
            .collect();

        if !openai_tools.is_empty() {
            openai_request.tools = Some(openai_tools);
        }
    }

    // Convert tool choice
    if let Some(ref tool_choice) = claude_request.tool_choice {
        if let Some(choice_type) = tool_choice.get("type").and_then(|v| v.as_str()) {
            openai_request.tool_choice = match choice_type {
                "auto" | "any" => Some(Value::String("auto".to_string())),
                "tool" => {
                    if let Some(name) = tool_choice.get("name").and_then(|v| v.as_str()) {
                        let mut choice_obj = HashMap::new();
                        choice_obj.insert("type".to_string(), Value::String(tool::FUNCTION.to_string()));

                        let mut function_obj = HashMap::new();
                        function_obj.insert("name".to_string(), Value::String(name.to_string()));
                        choice_obj.insert(tool::FUNCTION.to_string(), Value::Object(function_obj.into_iter().collect()));

                        Some(Value::Object(choice_obj.into_iter().collect()))
                    } else {
                        Some(Value::String("auto".to_string()))
                    }
                }
                _ => Some(Value::String("auto".to_string())),
            };
        }
    }

    debug!("Converted Claude request to OpenAI format");
    openai_request
}

/// Convert Claude user message to OpenAI format
fn convert_claude_user_message(msg: &ClaudeMessage) -> OpenAIMessage {
    match &msg.content {
        MessageContent::String(s) => OpenAIMessage {
            role: role::USER.to_string(),
            content: Some(Value::String(s.clone())),
            tool_calls: None,
            tool_call_id: None,
        },
        MessageContent::Blocks(blocks) => {
            // Handle multimodal content
            let mut openai_content = Vec::new();

            for block in blocks {
                match block {
                    ClaudeContentBlock::Text(text_block) => {
                        let mut content_obj = HashMap::new();
                        content_obj.insert("type".to_string(), Value::String("text".to_string()));
                        content_obj.insert("text".to_string(), Value::String(text_block.text.clone()));
                        openai_content.push(Value::Object(content_obj.into_iter().collect()));
                    }
                    ClaudeContentBlock::Image(image_block) => {
                        // Convert Claude image format to OpenAI format
                        if let (Some(source_type), Some(media_type), Some(data)) = (
                            image_block.source.get("type").and_then(|v| v.as_str()),
                            image_block.source.get("media_type").and_then(|v| v.as_str()),
                            image_block.source.get("data").and_then(|v| v.as_str()),
                        ) {
                            if source_type == "base64" {
                                let mut image_url_obj = HashMap::new();
                                image_url_obj.insert(
                                    "url".to_string(),
                                    Value::String(format!("data:{};base64,{}", media_type, data)),
                                );

                                let mut content_obj = HashMap::new();
                                content_obj.insert("type".to_string(), Value::String("image_url".to_string()));
                                content_obj.insert("image_url".to_string(), Value::Object(image_url_obj.into_iter().collect()));
                                openai_content.push(Value::Object(content_obj.into_iter().collect()));
                            }
                        }
                    }
                    _ => {}
                }
            }

            // If only one text block, return as simple string
            if openai_content.len() == 1 {
                if let Some(obj) = openai_content[0].as_object() {
                    if obj.get("type").and_then(|v| v.as_str()) == Some("text") {
                        if let Some(text) = obj.get("text") {
                            return OpenAIMessage {
                                role: role::USER.to_string(),
                                content: Some(text.clone()),
                                tool_calls: None,
                                tool_call_id: None,
                            };
                        }
                    }
                }
            }

            OpenAIMessage {
                role: role::USER.to_string(),
                content: Some(Value::Array(openai_content)),
                tool_calls: None,
                tool_call_id: None,
            }
        }
    }
}

/// Convert Claude assistant message to OpenAI format
fn convert_claude_assistant_message(msg: &ClaudeMessage) -> OpenAIMessage {
    match &msg.content {
        MessageContent::String(s) => OpenAIMessage {
            role: role::ASSISTANT.to_string(),
            content: Some(Value::String(s.clone())),
            tool_calls: None,
            tool_call_id: None,
        },
        MessageContent::Blocks(blocks) => {
            let mut text_parts = Vec::new();
            let mut tool_calls = Vec::new();

            for block in blocks {
                match block {
                    ClaudeContentBlock::Text(text_block) => {
                        text_parts.push(text_block.text.clone());
                    }
                    ClaudeContentBlock::ToolUse(tool_use) => {
                        tool_calls.push(OpenAIToolCall {
                            id: tool_use.id.clone(),
                            call_type: tool::FUNCTION.to_string(),
                            function: crate::models::openai::OpenAIFunction {
                                name: tool_use.name.clone(),
                                arguments: serde_json::to_string(&tool_use.input)
                                    .unwrap_or_else(|_| "{}".to_string()),
                            },
                        });
                    }
                    _ => {}
                }
            }

            let content = if text_parts.is_empty() {
                None
            } else {
                Some(Value::String(text_parts.join("")))
            };

            let tool_calls_opt = if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            };

            OpenAIMessage {
                role: role::ASSISTANT.to_string(),
                content,
                tool_calls: tool_calls_opt,
                tool_call_id: None,
            }
        }
    }
}

/// Convert Claude tool results to OpenAI format
fn convert_claude_tool_results(msg: &ClaudeMessage) -> Vec<OpenAIMessage> {
    let mut tool_messages = Vec::new();

    if let MessageContent::Blocks(blocks) = &msg.content {
        for block in blocks {
            if let ClaudeContentBlock::ToolResult(tool_result) = block {
                let content = parse_tool_result_content(&tool_result.content);
                tool_messages.push(OpenAIMessage {
                    role: role::TOOL.to_string(),
                    content: Some(Value::String(content)),
                    tool_calls: None,
                    tool_call_id: Some(tool_result.tool_use_id.clone()),
                });
            }
        }
    }

    tool_messages
}

/// Check if message has tool results
fn has_tool_results(msg: &ClaudeMessage) -> bool {
    if let MessageContent::Blocks(blocks) = &msg.content {
        blocks.iter().any(|block| matches!(block, ClaudeContentBlock::ToolResult(_)))
    } else {
        false
    }
}

/// Parse and normalize tool result content into a string format
fn parse_tool_result_content(content: &ToolResultContent) -> String {
    match content {
        ToolResultContent::String(s) => s.clone(),
        ToolResultContent::Array(arr) => {
            let parts: Vec<String> = arr
                .iter()
                .filter_map(|item| {
                    if let Some(text_type) = item.get("type").and_then(|v| v.as_str()) {
                        if text_type == content::TEXT {
                            return item.get("text").and_then(|v| v.as_str()).map(|s| s.to_string());
                        }
                    }
                    if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                        return Some(text.to_string());
                    }
                    serde_json::to_string(item).ok()
                })
                .collect();
            parts.join("\n").trim().to_string()
        }
        ToolResultContent::Object(obj) => {
            if let Some(text_type) = obj.get("type").and_then(|v| v.as_str()) {
                if text_type == content::TEXT {
                    if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                        return text.to_string();
                    }
                }
            }
            serde_json::to_string(obj).unwrap_or_else(|_| "{}".to_string())
        }
    }
}
