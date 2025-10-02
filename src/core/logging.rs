//! Logging configuration and initialization
//!
//! This module sets up the tracing subscriber for structured logging
//! throughout the application.

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Initialize the logging system with the specified level
///
/// Sets up tracing with a filter based on the provided log level.
/// If the log level is invalid, defaults to "info".
///
/// # Arguments
///
/// * `log_level` - The log level string (debug, info, warning, error, critical)
pub fn init_logging(log_level: &str) {
    // Parse log level - extract just the first word to handle comments
    let level = log_level
        .split_whitespace()
        .next()
        .unwrap_or("info")
        .to_lowercase();

    // Validate and set default if invalid
    let valid_levels = ["debug", "info", "warning", "warn", "error", "critical"];
    let final_level = if valid_levels.contains(&level.as_str()) {
        // Map "warning" to "warn" and "critical" to "error" for compatibility
        match level.as_str() {
            "warning" => "warn",
            "critical" => "error",
            other => other,
        }
    } else {
        "info"
    };

    // Create the environment filter
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(final_level));

    // Initialize the tracing subscriber
    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .init();
}
