//! Constants for API role and content types
//!
//! This module defines string constants used throughout the application for
//! message roles, content types, stop reasons, event types, and delta types.
//! Following JPL Rule 1: All identifiers use clear, descriptive names.

/// Message role constants
pub mod role {
    /// User role identifier
    pub const USER: &str = "user";

    /// Assistant role identifier
    pub const ASSISTANT: &str = "assistant";

    /// System role identifier
    pub const SYSTEM: &str = "system";

    /// Tool role identifier
    pub const TOOL: &str = "tool";
}

/// Content type constants
pub mod content {
    /// Text content type
    pub const TEXT: &str = "text";

    /// Image content type
    pub const IMAGE: &str = "image";

    /// Tool use content type
    pub const TOOL_USE: &str = "tool_use";

    /// Tool result content type
    pub const TOOL_RESULT: &str = "tool_result";
}

/// Tool type constants
pub mod tool {
    /// Function tool type
    pub const FUNCTION: &str = "function";
}

/// Stop reason constants
pub mod stop {
    /// End turn stop reason
    pub const END_TURN: &str = "end_turn";

    /// Max tokens stop reason
    pub const MAX_TOKENS: &str = "max_tokens";

    /// Tool use stop reason
    pub const TOOL_USE: &str = "tool_use";

    /// Error stop reason
    pub const ERROR: &str = "error";
}

/// Server-sent event type constants
pub mod event {
    /// Message start event
    pub const MESSAGE_START: &str = "message_start";

    /// Message stop event
    pub const MESSAGE_STOP: &str = "message_stop";

    /// Message delta event
    pub const MESSAGE_DELTA: &str = "message_delta";

    /// Content block start event
    pub const CONTENT_BLOCK_START: &str = "content_block_start";

    /// Content block stop event
    pub const CONTENT_BLOCK_STOP: &str = "content_block_stop";

    /// Content block delta event
    pub const CONTENT_BLOCK_DELTA: &str = "content_block_delta";

    /// Ping event
    pub const PING: &str = "ping";
}

/// Delta type constants
pub mod delta {
    /// Text delta type
    pub const TEXT: &str = "text_delta";

    /// Input JSON delta type
    pub const INPUT_JSON: &str = "input_json_delta";
}
