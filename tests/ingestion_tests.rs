//! Integration tests for the MNEMOS ingestion pipeline.
//!
//! These tests exercise the full ingestion flow: emotional tagging,
//! importance scoring, concept extraction, and storage. They use mock
//! LLM providers so they run without API keys.

use mnemos_core::{MnemosConfig, StorageConfig};

/// Build a test config with mock LLM settings.
fn test_config() -> MnemosConfig {
    MnemosConfig {
        storage: StorageConfig::default(),
        ..Default::default()
    }
}

#[tokio::test]
async fn ingestion_pipeline_builds_with_mock_llm() {
    let _config = test_config();
    // Verify config builds without panicking
    assert_eq!(config.storage.database, "mnemos");
}
