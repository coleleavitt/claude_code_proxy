//! Application configuration management
//!
//! This module handles loading and validating configuration from TOML files.
//! Following JPL Rule 24: All configuration is validated at startup.

use crate::core::provider::ProviderType;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Maximum token limit default
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// Minimum token limit default
const DEFAULT_MIN_TOKENS: u32 = 100;

/// Default request timeout in seconds
const DEFAULT_REQUEST_TIMEOUT: u64 = 90;

/// Default maximum retries
const DEFAULT_MAX_RETRIES: u32 = 2;

/// Default server port
const DEFAULT_PORT: u16 = 8082;

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAIConfig {
    pub api_key: String,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub azure_api_version: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenRouterConfig {
    pub api_key: String,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub site_url: Option<String>,
    #[serde(default)]
    pub app_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VertexAIConfig {
    pub project_id: String,
    pub location: String,
    pub access_token: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub big_model: String,
    pub middle_model: String,
    pub small_model: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RequestConfig {
    #[serde(default = "default_max_tokens")]
    pub max_tokens_limit: u32,
    #[serde(default = "default_min_tokens")]
    pub min_tokens_limit: u32,
    #[serde(default = "default_max_messages")]
    pub max_messages_limit: u32,
    #[serde(default = "default_request_timeout")]
    pub request_timeout: u64,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Maximum context tokens before compression
    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: u32,
    /// Target tokens after compression
    #[serde(default = "default_target_context_tokens")]
    pub target_context_tokens: u32,
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    DEFAULT_PORT
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_max_tokens() -> u32 {
    DEFAULT_MAX_TOKENS
}

fn default_min_tokens() -> u32 {
    DEFAULT_MIN_TOKENS
}

const DEFAULT_MAX_MESSAGES: u32 = 30;

fn default_max_messages() -> u32 {
    DEFAULT_MAX_MESSAGES
}

fn default_request_timeout() -> u64 {
    DEFAULT_REQUEST_TIMEOUT
}

fn default_max_retries() -> u32 {
    DEFAULT_MAX_RETRIES
}

/// Default maximum context tokens before compression (128K tokens)
const DEFAULT_MAX_CONTEXT_TOKENS: u32 = 128000;

/// Default target context tokens after compression (64K tokens)
const DEFAULT_TARGET_CONTEXT_TOKENS: u32 = 64000;

fn default_max_context_tokens() -> u32 {
    DEFAULT_MAX_CONTEXT_TOKENS
}

fn default_target_context_tokens() -> u32 {
    DEFAULT_TARGET_CONTEXT_TOKENS
}

#[derive(Debug, Clone, Deserialize)]
pub struct TomlConfig {
    pub provider: String,
    #[serde(default)]
    pub anthropic_api_key: Option<String>,
    #[serde(default)]
    pub openai: Option<OpenAIConfig>,
    #[serde(default)]
    pub openrouter: Option<OpenRouterConfig>,
    #[serde(default)]
    pub vertexai: Option<VertexAIConfig>,
    pub models: ModelConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub request: RequestConfig,
}

/// Application configuration loaded from TOML files
///
/// All configuration values are loaded and validated at startup to ensure
/// the application fails fast if misconfigured (JPL Rule 24).
#[derive(Debug, Clone)]
pub struct Config {
    /// Provider type (OpenAI, OpenRouter, or VertexAI)
    pub provider: ProviderType,

    /// OpenAI API key (required for OpenAI provider)
    pub openai_api_key: String,

    /// Optional Anthropic API key for client validation
    pub anthropic_api_key: Option<String>,

    /// OpenAI/OpenRouter API base URL
    pub openai_base_url: String,

    /// Azure API version (for Azure OpenAI deployments)
    pub azure_api_version: Option<String>,

    /// OpenRouter specific settings
    pub openrouter_site_url: Option<String>,
    pub openrouter_app_name: Option<String>,

    /// Vertex AI specific settings
    pub vertexai_project_id: Option<String>,
    pub vertexai_location: Option<String>,
    pub vertexai_access_token: Option<String>,

    /// Server host address
    pub host: String,

    /// Server port
    pub port: u16,

    /// Logging level
    pub log_level: String,

    /// Maximum tokens limit
    pub max_tokens_limit: u32,

    /// Message limit for context truncation
    pub max_messages_limit: u32,

    /// Minimum tokens limit
    pub min_tokens_limit: u32,

    /// Request timeout in seconds
    pub request_timeout: u64,

    /// Maximum number of retries
    pub max_retries: u32,

    /// Maximum context tokens before compression
    pub max_context_tokens: u32,

    /// Target context tokens after compression
    pub target_context_tokens: u32,

    /// Model for opus requests
    pub big_model: String,

    /// Model for sonnet requests
    pub middle_model: String,

    /// Model for haiku requests
    pub small_model: String,
}

impl Config {
    /// Load configuration from TOML file
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - The TOML file cannot be read or parsed
    /// - Required configuration values are missing
    /// - Configuration values are invalid
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path).context("Failed to read configuration file")?;

        let config: TomlConfig =
            toml::from_str(&content).context("Failed to parse TOML configuration")?;

        let provider = ProviderType::from_str(&config.provider)
            .context("Invalid provider value. Must be one of: openai, openrouter, vertexai")?;

        let (
            openai_api_key,
            openai_base_url,
            azure_api_version,
            openrouter_site_url,
            openrouter_app_name,
        ) = match provider {
            ProviderType::OpenAI => {
                let openai_config = config
                    .openai
                    .context("OpenAI configuration missing for OpenAI provider")?;
                let azure_ver = openai_config.azure_api_version.clone();
                (
                    openai_config.api_key,
                    openai_config
                        .base_url
                        .unwrap_or_else(|| "https://api.openai.com/v1".to_string()),
                    azure_ver,
                    None,
                    None,
                )
            }
            ProviderType::OpenRouter => {
                let openrouter_config = config
                    .openrouter
                    .context("OpenRouter configuration missing for OpenRouter provider")?;
                (
                    openrouter_config.api_key,
                    openrouter_config
                        .base_url
                        .unwrap_or_else(|| "https://openrouter.ai/api/v1".to_string()),
                    None,
                    openrouter_config.site_url,
                    openrouter_config.app_name,
                )
            }
            ProviderType::VertexAI => {
                (String::new(), String::new(), None, None, None) // Not used for Vertex AI
            }
        };

        let vertexai_config = if provider == ProviderType::VertexAI {
            let config = config
                .vertexai
                .context("Vertex AI configuration missing for Vertex AI provider")?;
            (
                Some(config.project_id),
                Some(config.location),
                Some(config.access_token),
            )
        } else {
            (None, None, None)
        };

        Ok(Config {
            provider,
            openai_api_key,
            anthropic_api_key: config.anthropic_api_key,
            openai_base_url,
            azure_api_version,
            openrouter_site_url,
            openrouter_app_name,
            vertexai_project_id: vertexai_config.0,
            vertexai_location: vertexai_config.1,
            vertexai_access_token: vertexai_config.2,
            host: config.server.host,
            port: config.server.port,
            log_level: config.server.log_level,
            max_tokens_limit: config.request.max_tokens_limit,
            min_tokens_limit: config.request.min_tokens_limit,
            max_messages_limit: config.request.max_messages_limit,
            request_timeout: config.request.request_timeout,
            max_retries: config.request.max_retries,
            max_context_tokens: config.request.max_context_tokens,
            target_context_tokens: config.request.target_context_tokens,
            big_model: config.models.big_model,
            middle_model: config.models.middle_model,
            small_model: config.models.small_model,
        })
    }

    /// Load configuration from environment and config file
    ///
    /// Looks for config.toml in current directory by default
    pub fn from_env() -> Result<Self> {
        let config_path =
            std::env::var("CONFIG_PATH").unwrap_or_else(|_| "config.toml".to_string());
        Self::from_file(config_path)
    }

    /// Validate API key format based on provider
    ///
    /// For OpenAI: checks that the API key starts with 'sk-' prefix
    /// For other providers: checks that key is non-empty
    pub fn validate_api_key(&self) -> bool {
        match self.provider {
            ProviderType::OpenAI => {
                !self.openai_api_key.is_empty() && self.openai_api_key.starts_with("sk-")
            }
            ProviderType::OpenRouter => !self.openai_api_key.is_empty(),
            ProviderType::VertexAI => {
                self.vertexai_access_token.is_some() && self.vertexai_project_id.is_some()
            }
        }
    }

    /// Validate client's Anthropic API key
    ///
    /// If anthropic_api_key is set, validates that client_api_key matches.
    /// If not set, validation is skipped and returns true.
    pub fn validate_client_api_key(&self, client_api_key: &str) -> bool {
        match &self.anthropic_api_key {
            Some(expected_key) => client_api_key == expected_key,
            None => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_config() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        write!(
            file,
            r#"
            provider = "openai"
            anthropic_api_key = "test-key"

            [openai]
            api_key = "sk-test123"
            base_url = "https://api.openai.com/v1"

            [models]
            big_model = "gpt-4o"
            middle_model = "gpt-4o"
            small_model = "gpt-4o-mini"

            [server]
            host = "0.0.0.0"
            port = 8082
            log_level = "info"

            [request]
            max_tokens_limit = 4096
            min_tokens_limit = 100
            max_messages_limit = 30
            request_timeout = 90
            max_retries = 2
            max_context_tokens = 120000
            target_context_tokens = 80000
        "#
        )
        .unwrap();
        file.flush().unwrap();
        file
    }

    #[test]
    fn test_load_config() {
        let file = create_test_config();
        let config = Config::from_file(file.path()).unwrap();
        assert_eq!(config.provider, ProviderType::OpenAI);
        assert_eq!(config.openai_api_key, "sk-test123");
        assert_eq!(config.anthropic_api_key, Some("test-key".to_string()));
    }

    #[test]
    fn test_validate_api_key() {
        let file = create_test_config();
        let config = Config::from_file(file.path()).unwrap();
        assert!(config.validate_api_key());
    }

    #[test]
    fn test_validate_client_api_key() {
        let file = create_test_config();
        let config = Config::from_file(file.path()).unwrap();
        assert!(config.validate_client_api_key("test-key"));
        assert!(!config.validate_client_api_key("wrong-key"));
    }
}
