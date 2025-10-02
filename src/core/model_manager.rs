//! Model mapping between Claude and OpenAI model names
//!
//! This module handles the conversion of Claude model names to their
//! corresponding OpenAI model equivalents based on configuration.

use crate::core::config::Config;

/// Manages model name mapping from Claude to OpenAI
pub struct ModelManager {
    config: Config,
}

impl ModelManager {
    /// Create a new ModelManager with the given configuration
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Map Claude model names to OpenAI model names
    ///
    /// Maps Claude model names (opus, sonnet, haiku) to configured OpenAI
    /// models. If the model name is already an OpenAI model or other supported
    /// model (ARK, Doubao, DeepSeek), returns it as-is.
    ///
    /// # Arguments
    ///
    /// * `claude_model` - The Claude model name to map
    ///
    /// # Returns
    ///
    /// The corresponding OpenAI model name
    pub fn map_claude_model_to_openai(&self, claude_model: &str) -> String {
        // If it's already an OpenAI model, return as-is
        if claude_model.starts_with("gpt-") || claude_model.starts_with("o1-") {
            return claude_model.to_string();
        }

        // If it's other supported models (ARK/Doubao/DeepSeek), return as-is
        if claude_model.starts_with("ep-")
            || claude_model.starts_with("doubao-")
            || claude_model.starts_with("deepseek-")
        {
            return claude_model.to_string();
        }

        // Map based on model naming patterns
        let model_lower = claude_model.to_lowercase();
        if model_lower.contains("haiku") {
            self.config.small_model.clone()
        } else if model_lower.contains("sonnet") {
            self.config.middle_model.clone()
        } else if model_lower.contains("opus") {
            self.config.big_model.clone()
        } else {
            // Default to big model for unknown models
            self.config.big_model.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::provider::ProviderType;

    fn create_test_config() -> Config {
        Config {
            provider: ProviderType::OpenAI,
            openai_api_key: "sk-test".to_string(),
            anthropic_api_key: None,
            openai_base_url: "https://api.openai.com/v1".to_string(),
            azure_api_version: None,
            openrouter_site_url: None,
            openrouter_app_name: None,
            vertexai_project_id: None,
            vertexai_location: None,
            vertexai_access_token: None,
            host: "0.0.0.0".to_string(),
            port: 8082,
            log_level: "INFO".to_string(),
            max_tokens_limit: 4096,
            min_tokens_limit: 100,
            request_timeout: 90,
            max_retries: 2,
            big_model: "gpt-4o".to_string(),
            middle_model: "gpt-4o".to_string(),
            small_model: "gpt-4o-mini".to_string(),
        }
    }

    #[test]
    fn test_map_haiku_model() {
        let manager = ModelManager::new(create_test_config());
        assert_eq!(
            manager.map_claude_model_to_openai("claude-3-haiku-20240307"),
            "gpt-4o-mini"
        );
    }

    #[test]
    fn test_map_sonnet_model() {
        let manager = ModelManager::new(create_test_config());
        assert_eq!(
            manager.map_claude_model_to_openai("claude-3-5-sonnet-20241022"),
            "gpt-4o"
        );
    }

    #[test]
    fn test_map_opus_model() {
        let manager = ModelManager::new(create_test_config());
        assert_eq!(
            manager.map_claude_model_to_openai("claude-3-opus-20240229"),
            "gpt-4o"
        );
    }

    #[test]
    fn test_passthrough_openai_model() {
        let manager = ModelManager::new(create_test_config());
        assert_eq!(
            manager.map_claude_model_to_openai("gpt-4-turbo"),
            "gpt-4-turbo"
        );
    }
}
