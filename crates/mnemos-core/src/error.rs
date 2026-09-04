//! Error type shared by all crates.

/// All MNEMOS failures funnel through here.
#[derive(Debug, thiserror::Error)]
pub enum MnemosError {
    #[error("storage: {0}")]
    Storage(String),
    #[error("llm: {0}")]
    Llm(String),
    #[error("embedding: {0}")]
    Embedding(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("config: {0}")]
    Config(String),
    #[error("serialization: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("http: {0}")]
    Http(String),
    #[error("internal: {0}")]
    Internal(String),
}

/// Local alias so signatures stay short.
pub type Result<T> = std::result::Result<T, MnemosError>;
