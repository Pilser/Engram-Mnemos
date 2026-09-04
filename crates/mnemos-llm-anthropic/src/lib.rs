//! Anthropic chat provider (OpenAI-compatible endpoint).
//!
//! Thin wrapper around [`OpenAiCompatibleProvider`] preconfigured for
//! Anthropic's OpenAI-compatible base URL (`https://api.anthropic.com/v1`).
//!
//! NOTE: the true Anthropic wire format (`x-api-key` header against
//! `/v1/messages`) is a later enhancement. Per `PROJECT-PLAN.md`, this crate
//! uses the OpenAI-compatible endpoint (`/chat/completions` with bearer auth)
//! via the shared [`OpenAiCompatibleProvider`] until native message-format
//! support lands.

use async_trait::async_trait;
use mnemos_core::{LlmConfig, Result};
use mnemos_llm_openai::OpenAiCompatibleProvider;
use mnemos_llm_trait::LlmProvider;

/// Anthropic's default OpenAI-compatible base URL.
pub const ANTHROPIC_DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";

/// Chat provider backed by Anthropic's OpenAI-compatible endpoint.
pub struct AnthropicProvider {
    inner: OpenAiCompatibleProvider,
}

impl AnthropicProvider {
    /// Build for `model` against the Anthropic default endpoint.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            inner: OpenAiCompatibleProvider::new(
                model,
                ANTHROPIC_DEFAULT_BASE_URL,
                api_key,
            ),
        }
    }

    /// Build from unified [`LlmConfig`], using `config.api_key` + `config.model`.
    ///
    /// NOTE: `config.base_url` is intentionally ignored — it defaults to the
    /// `OpenAI` endpoint (`https://api.openai.com/v1` via `LlmConfig::from_env`)
    /// and must never leak into Anthropic traffic. Anthropic always targets
    /// its own default (`https://api.anthropic.com/v1`).
    #[must_use]
    pub fn from_config(config: &LlmConfig) -> Self {
        Self::new(config.api_key.clone(), config.model.clone())
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
impl LlmProvider for AnthropicProvider {
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
    fn new_uses_anthropic_base_url() {
        let p = AnthropicProvider::new("sk-ant-test", "claude-sonnet-4-5");
        assert_eq!(p.model(), "claude-sonnet-4-5");
        assert_eq!(p.base_url(), ANTHROPIC_DEFAULT_BASE_URL);
        assert_eq!(p.base_url(), "https://api.anthropic.com/v1");
    }

    #[test]
    fn from_config_maps_model_and_ignores_base_url() {
        let cfg = LlmConfig {
            api_key: "sk-ant-test".into(),
            base_url: "https://api.openai.com/v1".into(),
            model: "claude-sonnet-4-5".into(),
            embedding_model: "emb".into(),
            embedding_base_url: "https://api.openai.com/v1".into(),
        };
        let p = AnthropicProvider::from_config(&cfg);
        assert_eq!(p.model(), "claude-sonnet-4-5");
        assert_eq!(p.base_url(), ANTHROPIC_DEFAULT_BASE_URL);
    }

    #[tokio::test]
    #[ignore = "needs a live Anthropic endpoint + ANTHROPIC_API_KEY"]
    async fn live_chat_smoke() {
        let key = std::env::var("ANTHROPIC_API_KEY").unwrap_or_default();
        let p = AnthropicProvider::new(key, "claude-sonnet-4-5");
        let _ = p.chat("ping").await;
    }
}
