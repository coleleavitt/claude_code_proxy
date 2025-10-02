//! Claude API data models
//!
//! This module defines the request and response structures for the Claude API,
//! matching the Anthropic Messages API format.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Text content block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeContentBlockText {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

/// Image content block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeContentBlockImage {
    #[serde(rename = "type")]
    pub content_type: String,
    pub source: HashMap<String, serde_json::Value>,
}

/// Tool use content block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeContentBlockToolUse {
    #[serde(rename = "type")]
    pub content_type: String,
    pub id: String,
    pub name: String,
    pub input: HashMap<String, serde_json::Value>,
}

/// Tool result content block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeContentBlockToolResult {
    #[serde(rename = "type")]
    pub content_type: String,
    pub tool_use_id: String,
    #[serde(with = "tool_result_content")]
    pub content: ToolResultContent,
}

/// Tool result content can be a string, list of objects, or a single object
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    String(String),
    Array(Vec<HashMap<String, serde_json::Value>>),
    Object(HashMap<String, serde_json::Value>),
}

mod tool_result_content {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(content: &ToolResultContent, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match content {
            ToolResultContent::String(s) => serializer.serialize_str(s),
            ToolResultContent::Array(a) => a.serialize(serializer),
            ToolResultContent::Object(o) => o.serialize(serializer),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<ToolResultContent, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: serde_json::Value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::String(s) => Ok(ToolResultContent::String(s)),
            serde_json::Value::Array(a) => {
                let objects: Vec<HashMap<String, serde_json::Value>> = a
                    .into_iter()
                    .filter_map(|v| {
                        if let serde_json::Value::Object(m) = v {
                            Some(m.into_iter().collect())
                        } else {
                            None
                        }
                    })
                    .collect();
                Ok(ToolResultContent::Array(objects))
            }
            serde_json::Value::Object(o) => Ok(ToolResultContent::Object(o.into_iter().collect())),
            _ => Err(serde::de::Error::custom("Invalid tool result content")),
        }
    }
}

/// Union type for different content block types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ClaudeContentBlock {
    Text(ClaudeContentBlockText),
    Image(ClaudeContentBlockImage),
    ToolUse(ClaudeContentBlockToolUse),
    ToolResult(ClaudeContentBlockToolResult),
}

/// System content block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeSystemContent {
    #[serde(rename = "type")]
    pub content_type: String,
    pub text: String,
}

/// Message with role and content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeMessage {
    pub role: String,
    #[serde(with = "message_content")]
    pub content: MessageContent,
}

/// Message content can be a string or array of content blocks
#[derive(Debug, Clone)]
pub enum MessageContent {
    String(String),
    Blocks(Vec<ClaudeContentBlock>),
}

mod message_content {
    use super::*;
    use serde::{Deserializer, Serializer};

    pub fn serialize<S>(content: &MessageContent, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match content {
            MessageContent::String(s) => serializer.serialize_str(s),
            MessageContent::Blocks(b) => b.serialize(serializer),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<MessageContent, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: serde_json::Value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::String(s) => Ok(MessageContent::String(s)),
            serde_json::Value::Array(_) => {
                let blocks: Vec<ClaudeContentBlock> =
                    serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(MessageContent::Blocks(blocks))
            }
            _ => Err(serde::de::Error::custom("Invalid message content")),
        }
    }
}

/// Tool definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: HashMap<String, serde_json::Value>,
}

/// Thinking configuration for extended thinking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeThinkingConfig {
    #[serde(default = "default_thinking_enabled")]
    pub enabled: bool,
}

fn default_thinking_enabled() -> bool {
    true
}

/// System content can be a string or array of system content blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SystemContent {
    String(String),
    Blocks(Vec<ClaudeSystemContent>),
}

/// Claude Messages API request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeMessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<ClaudeMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ClaudeTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<HashMap<String, serde_json::Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ClaudeThinkingConfig>,
}

fn default_temperature() -> f32 {
    1.0
}

/// Claude token count request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeTokenCountRequest {
    pub model: String,
    pub messages: Vec<ClaudeMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<SystemContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ClaudeTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ClaudeThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<HashMap<String, serde_json::Value>>,
}
