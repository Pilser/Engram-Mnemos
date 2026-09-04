//! Local LLM via Ollama (OpenAI-compatible API).
//!
//! Thin wrapper around [`OpenAiCompatibleProvider`] preconfigured for
//! Ollama's default endpoint (`http://localhost:11434/v1`).

use async_trait::async_trait;
use mnemos_core::{LlmConfig, Result};
use mnemos_llm_openai::OpenAiCompatibleProvider;
use mnemos_llm_trait::LlmProvider;

/// Ollama's default OpenAI-compatible base URL.
pub const OLLAMA_DEFAULT_BASE_URL: &str = "http://localhost:11434/v1";

/// Chat provider backed by a local Ollama server.
pub struct OllamaProvider {
    inner: OpenAiCompatibleProvider,
}

impl OllamaProvider {
    /// Build for `model` against the default Ollama endpoint with no API key.
    pub fn new(model: impl Into<String>) -> Self {
        Self::with_base(model, OLLAMA_DEFAULT_BASE_URL)
    }

    /// Build for `model` against a custom Ollama host with no API key.
    pub fn with_base(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            inner: OpenAiCompatibleProvider::new(model, base_url, String::new()),
        }
    }

    /// Build from unified [`LlmConfig`], using only `config.model`.
    ///
    /// NOTE: `config.base_url` is intentionally ignored — Ollama always
    /// targets its local default (`http://localhost:11434/v1`). Use
    /// [`OllamaProvider::with_base`] for custom hosts.
    #[must_use]
    pub fn from_config(config: &LlmConfig) -> Self {
        Self::new(config.model.clone())
    }

    /// Model sent in the `model` field.
    #[must_use]
    pub fn model(&self) -> &str {
        self.inner.model()
    }

    /// Normalized base URL (no trailing slash).
    #[must_use]
    pub fn base_url(&self) -> &str {
        self.inner.base_url()
    }
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    async fn chat(&self, prompt: &str) -> Result<String> {
        self.inner.chat(prompt).await
    }

    async fn chat_with_system(&self, system: &str, user: &str) -> Result<String> {
        self.inner.chat_with_system(system, user).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_uses_default_base_url() {
        let p = OllamaProvider::new("llama3.1");
        assert_eq!(p.model(), "llama3.1");
        assert_eq!(p.base_url(), OLLAMA_DEFAULT_BASE_URL);
    }

    #[test]
    fn with_base_maps_model_and_trims_slash() {
        let p = OllamaProvider::with_base("qwen2.5", "http://ollama.lan:11434/v1/");
        assert_eq!(p.model(), "qwen2.5");
        assert_eq!(p.base_url(), "http://ollama.lan:11434/v1");
    }

    #[test]
    fn from_config_uses_model_and_ignores_base_url() {
        let cfg = LlmConfig {
            api_key: "sk-test".into(),
            base_url: "https://api.openai.com/v1".into(),
            model: "llama3.1".into(),
            embedding_model: "emb".into(),
            embedding_base_url: "https://api.openai.com/v1".into(),
            embedding_dim: 1536,
        };
        let p = OllamaProvider::from_config(&cfg);
        assert_eq!(p.model(), "llama3.1");
        assert_eq!(p.base_url(), OLLAMA_DEFAULT_BASE_URL);
    }

    #[tokio::test]
    #[ignore = "needs Ollama running at http://localhost:11434"]
    async fn live_chat_smoke() {
        let p = OllamaProvider::new("llama3.1");
        let _ = p.chat("ping").await;
    }
}
