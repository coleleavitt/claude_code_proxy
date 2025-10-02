//! Claude-to-OpenAI API Proxy
//!
//! This application acts as a proxy that accepts Claude API requests and
//! forwards them to OpenAI API endpoints, converting between the two formats.

mod api;
mod conversion;
mod core;
mod models;

use crate::api::endpoints::{create_router, AppState};
use crate::core::config::Config;
use crate::core::logging::init_logging;
use crate::core::model_manager::ModelManager;
use crate::core::provider::{Provider, ProviderType};
use crate::core::providers::{OpenAIProvider, OpenRouterProvider, VertexAIProvider};
use std::sync::Arc;
use tracing::{error, info};

#[tokio::main]
async fn main() {
    // Check for --help flag
    if std::env::args().any(|arg| arg == "--help") {
        print_help();
        return;
    }

    // Load configuration
    let config = match Config::from_env() {
        Ok(cfg) => Arc::new(cfg),
        Err(e) => {
            eprintln!("Configuration Error: {}", e);
            std::process::exit(1);
        }
    };

    // Initialize logging
    init_logging(&config.log_level);

    // Print startup banner
    print_startup_banner(&config);

    // Validate API key
    if !config.validate_api_key() {
        error!("Invalid API key configuration for provider: {:?}", config.provider);
        std::process::exit(1);
    }

    // Create model manager
    let model_manager = Arc::new(ModelManager::new((*config).clone()));

    // Create provider based on configuration
    let provider: Arc<dyn Provider> = match config.provider {
        ProviderType::OpenAI => Arc::new(OpenAIProvider::new(
            config.openai_api_key.clone(),
            config.openai_base_url.clone(),
            config.request_timeout,
            config.azure_api_version.clone(),
        )),
        ProviderType::OpenRouter => Arc::new(OpenRouterProvider::new(
            config.openai_api_key.clone(),
            Some(config.openai_base_url.clone()),
            config.request_timeout,
            config.openrouter_site_url.clone(),
            config.openrouter_app_name.clone(),
        )),
        ProviderType::VertexAI => Arc::new(VertexAIProvider::new(
            config.vertexai_project_id.clone().unwrap(),
            config.vertexai_location.clone().unwrap(),
            config.vertexai_access_token.clone().unwrap(),
            config.request_timeout,
        )),
    };

    info!("Using provider: {}", provider.provider_name());

    // Create application state
    let app_state = AppState {
        config: config.clone(),
        model_manager,
        provider,
    };

    // Create router
    let app = create_router(app_state);

    // Bind to address
    let addr = format!("{}:{}", config.host, config.port);
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(listener) => listener,
        Err(e) => {
            error!("Failed to bind to {}: {}", addr, e);
            std::process::exit(1);
        }
    };

    info!("Server listening on http://{}", addr);

    // Run server
    if let Err(e) = axum::serve(listener, app).await {
        error!("Server error: {}", e);
        std::process::exit(1);
    }
}

/// Print startup banner with configuration
fn print_startup_banner(config: &Config) {
    println!("🚀 Claude API Proxy v1.0.0");
    println!("✅ Configuration loaded successfully");
    println!("   Provider: {:?}", config.provider);
    if !config.openai_base_url.is_empty() {
        println!("   Base URL: {}", config.openai_base_url);
    }
    println!("   Big Model (opus): {}", config.big_model);
    println!("   Middle Model (sonnet): {}", config.middle_model);
    println!("   Small Model (haiku): {}", config.small_model);
    println!("   Max Tokens Limit: {}", config.max_tokens_limit);
    println!("   Request Timeout: {}s", config.request_timeout);
    println!("   Server: {}:{}", config.host, config.port);
    println!(
        "   Client API Key Validation: {}",
        if config.anthropic_api_key.is_some() {
            "Enabled"
        } else {
            "Disabled"
        }
    );
    println!();
}

/// Print help message
fn print_help() {
    println!("Claude API Proxy v1.0.0");
    println!();
    println!("Usage: claude-code-proxy [OPTIONS]");
    println!();
    println!("Options:");
    println!("  --help    Display this help message");
    println!();
    println!("Environment variables:");
    println!("  PROVIDER - Provider type: openai, openrouter, vertexai (default: openai)");
    println!();
    println!("OpenAI/OpenRouter provider:");
    println!("  OPENAI_API_KEY / OPENROUTER_API_KEY - Your API key (required)");
    println!("  OPENAI_BASE_URL / OPENROUTER_BASE_URL - API base URL");
    println!("  AZURE_API_VERSION - Azure API version (for Azure OpenAI)");
    println!("  OPENROUTER_SITE_URL - Site URL for OpenRouter credits");
    println!("  OPENROUTER_APP_NAME - Application name for OpenRouter");
    println!();
    println!("Vertex AI provider:");
    println!("  VERTEXAI_PROJECT_ID - Google Cloud project ID (required)");
    println!("  VERTEXAI_LOCATION - GCP location (default: us-central1)");
    println!("  VERTEXAI_ACCESS_TOKEN - GCP access token (required)");
    println!();
    println!("Common settings:");
    println!("  ANTHROPIC_API_KEY - Expected Anthropic API key for client validation");
    println!("  BIG_MODEL - Model for opus requests (default: gpt-4o)");
    println!("  MIDDLE_MODEL - Model for sonnet requests (default: gpt-4o)");
    println!("  SMALL_MODEL - Model for haiku requests (default: gpt-4o-mini)");
    println!("  HOST - Server host (default: 0.0.0.0)");
    println!("  PORT - Server port (default: 8082)");
    println!("  LOG_LEVEL - Logging level (default: info)");
    println!("  MAX_TOKENS_LIMIT - Token limit (default: 4096)");
    println!("  MIN_TOKENS_LIMIT - Minimum token limit (default: 100)");
    println!("  REQUEST_TIMEOUT - Request timeout in seconds (default: 90)");
    println!();
    println!("Model mapping:");
    println!("  Claude haiku models -> SMALL_MODEL");
    println!("  Claude sonnet models -> MIDDLE_MODEL");
    println!("  Claude opus models -> BIG_MODEL");
}
