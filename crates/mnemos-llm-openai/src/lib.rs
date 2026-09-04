//! `OpenAI`-compatible chat provider (`OpenAI`, `Groq`, `Together`, `Ollama`).
//!
//! POSTs to `{base_url}/chat/completions` (Chat Completions API).

use async_trait::async_trait;
use mnemos_core::{LlmConfig, MnemosError, Result};

/// Chat client for any OpenAI-compatible `/chat/completions` endpoint.
pub struct OpenAiCompatibleProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl OpenAiCompatibleProvider {
    /// Build from explicit parts; trailing `/`s are stripped from `base_url`.
    pub fn new(
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            model: model.into(),
        }
    }

    /// Build from unified [`LlmConfig`].
    #[must_use]
    pub fn from_config(config: &LlmConfig) -> Self {
        Self::new(config.model.clone(), config.base_url.clone(), config.api_key.clone())
    }

    /// Model sent in the `model` field.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Normalized base URL (no trailing slash).
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Full chat-completions endpoint URL.
    fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    /// POST `messages` and return the assistant text.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Http`] on transport/JSON failures and
    /// [`MnemosError::Llm`] on non-2xx status or an unparsable body.
    async fn complete(&self, messages: serde_json::Value) -> Result<String> {
        self.complete_with_json_mode(messages, false).await
    }

    /// POST `messages`, optionally constraining decoding to a JSON object.
    ///
    /// `json_mode` sets `response_format: {"type": "json_object"}` so the
    /// model cannot think aloud when the caller needs machine parsing.
    /// Non-OpenAI-compatible backends may ignore the flag (then the
    /// trait-level repair loop is the backstop).
    async fn complete_with_json_mode(
        &self,
        messages: serde_json::Value,
        json_mode: bool,
    ) -> Result<String> {
        let start = std::time::Instant::now();
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        });
        if json_mode {
            body["response_format"] = serde_json::json!({"type": "json_object"});
        }
        let mut req = self.client.post(self.chat_url()).json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let out = self.send_and_extract(req).await;
        mnemos_telemetry::global().record_with_latency(
            "mnemos-llm-openai",
            if json_mode { "llm.chat_json" } else { "llm.chat" },
            out.is_ok(),
            &out.as_ref().map_or_else(std::string::ToString::to_string, |s| {
                format!("{} chars", s.len())
            }),
            elapsed_ms(start),
        );
        out
    }

    /// Send a prepared request and extract the assistant text.
    ///
    /// Retries transient failures (network errors, 429, 5xx) with
    /// exponential backoff. Client errors (400, 401, 403, 404) are not
    /// retried. Max retries configurable via `MNEMOS_LLM_MAX_RETRIES`
    /// (default 3).
    async fn send_and_extract(
        &self,
        req: reqwest::RequestBuilder,
    ) -> Result<String> {
        let max_retries = max_retries();
        let mut last_err = String::new();
        for attempt in 0..=max_retries {
            // Clone the request body for retry (RequestBuilder can't be
            // reused after sending). We rebuild from the same JSON each time.
            let req = match req.try_clone() {
                Some(clone) => clone,
                None => {
                    return Err(MnemosError::Http(
                        "request body not cloneable (streaming?)".to_string(),
                    ));
                }
            };
            match req.send().await {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        let json: serde_json::Value = resp
                            .json()
                            .await
                            .map_err(|e| MnemosError::Http(e.to_string()))?;
                        return extract_content(&json);
                    }
                    // Retry only on transient server errors / rate limits.
                    if status.is_server_error()
                        || status.as_u16() == 429
                        || status.as_u16() == 408
                    {
                        last_err = format!("HTTP {status}");
                        if attempt < max_retries {
                            tokio::time::sleep(backoff(attempt)).await;
                            continue;
                        }
                        return Err(MnemosError::Llm(format!(
                            "HTTP {status} after {attempt} retries"
                        )));
                    }
                    // Client error — don't retry.
                    let json: serde_json::Value = resp
                        .json()
                        .await
                        .map_err(|e| MnemosError::Http(e.to_string()))?;
                    return Err(MnemosError::Llm(format!("HTTP {status}: {json}")));
                }
                Err(e) => {
                    last_err = e.to_string();
                    if attempt < max_retries && is_retriable_error(&e) {
                        tokio::time::sleep(backoff(attempt)).await;
                        continue;
                    }
                    return Err(MnemosError::Http(format!(
                        "transport error after {attempt} retries: {last_err}"
                    )));
                }
            }
        }
        Err(MnemosError::Http(format!(
            "exhausted retries: {last_err}"
        )))
    }
}

/// Max retry attempts from `MNEMOS_LLM_MAX_RETRIES` (default 3).
fn max_retries() -> usize {
    parse_max_retries(std::env::var("MNEMOS_LLM_MAX_RETRIES").ok().as_deref())
}

/// Pure parser behind [`max_retries`] (kept side-effect free so tests never
/// touch the shared process env — env-mutating tests race in parallel).
fn parse_max_retries(raw: Option<&str>) -> usize {
    raw.and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(3)
}

/// Exponential backoff: 100ms * 2^attempt, capped at 2s.
fn backoff(attempt: usize) -> std::time::Duration {
    let ms = 100_u64.saturating_mul(2_u64.saturating_pow(attempt as u32));
    std::time::Duration::from_millis(ms.min(2000))
}

/// True for transport-level errors worth retrying (timeout, connect, etc.).
fn is_retriable_error(error: &reqwest::Error) -> bool {
    error.is_timeout() || error.is_connect() || error.is_request()
}

/// Pull `choices[0].message.content` out of a Chat Completions body.
///
/// # Errors
///
/// Returns [`MnemosError::Llm`] when the content string is absent.
fn extract_content(json: &serde_json::Value) -> Result<String> {
    json.pointer("/choices/0/message/content")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| MnemosError::Llm(format!("missing choices[0].message.content: {json}")))
}

#[async_trait]
impl mnemos_llm_trait::LlmProvider for OpenAiCompatibleProvider {
    async fn chat(&self, prompt: &str) -> Result<String> {
        self.complete(serde_json::json!([{ "role": "user", "content": prompt }])).await
    }

    async fn chat_with_system(&self, system: &str, user: &str) -> Result<String> {
        self.complete(serde_json::json!([
            { "role": "system", "content": system },
            { "role": "user", "content": user },
        ]))
        .await
    }
}

#[async_trait]
impl mnemos_llm_trait::StructuredOutput for OpenAiCompatibleProvider {
    /// Constrained decoding: `response_format json_object`, then parse.
    async fn chat_json(&self, prompt: &str) -> Result<serde_json::Value> {
        let raw = self
            .complete_with_json_mode(
                serde_json::json!([{ "role": "user", "content": prompt }]),
                true,
            )
            .await?;
        serde_json::from_str(&raw).map_err(|e| {
            MnemosError::Llm(format!("json_mode reply was not JSON: {e}; reply: {raw}"))
        })
    }
}

/// Elapsed milliseconds, saturating.
fn elapsed_ms(start: std::time::Instant) -> u64 {
    start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnemos_llm_trait::LlmProvider;

    #[test]
    fn url_join_has_no_trailing_slash_duplication() {
        for base in ["https://api.openai.com/v1", "https://api.openai.com/v1/", "http://x///"] {
            let p = OpenAiCompatibleProvider::new("m", base, "k");
            assert!(!p.base_url().ends_with('/'), "{base}");
            assert!(p.chat_url().ends_with("/chat/completions"));
            assert!(!p.chat_url().contains("//chat"));
        }
    }

    #[test]
    fn from_config_maps_fields() {
        let cfg = LlmConfig {
            api_key: "k".into(),
            base_url: "http://localhost:11434/v1/".into(),
            model: "llama".into(),
            embedding_model: "emb".into(),
            embedding_base_url: "http://localhost:11434/v1".into(),
        };
        let p = OpenAiCompatibleProvider::from_config(&cfg);
        assert_eq!(p.model(), "llama");
        assert_eq!(p.base_url(), "http://localhost:11434/v1");
    }

    #[test]
    fn empty_choices_yields_err() {
        let json = serde_json::json!({ "choices": [] });
        assert!(extract_content(&json).is_err());
        let json = serde_json::json!({ "id": "x" });
        assert!(extract_content(&json).is_err());
    }

    #[test]
    fn backoff_doubles_and_caps() {
        assert_eq!(backoff(0).as_millis(), 100);
        assert_eq!(backoff(1).as_millis(), 200);
        assert_eq!(backoff(2).as_millis(), 400);
        assert_eq!(backoff(3).as_millis(), 800);
        assert_eq!(backoff(4).as_millis(), 1600);
        // Capped at 2000ms.
        assert_eq!(backoff(10).as_millis(), 2000);
    }

    #[test]
    fn max_retries_defaults_to_three() {
        assert_eq!(parse_max_retries(None), 3);
        assert_eq!(parse_max_retries(Some("")), 3);
        assert_eq!(parse_max_retries(Some("not-a-number")), 3);
    }

    #[test]
    fn max_retries_reads_env() {
        assert_eq!(parse_max_retries(Some("5")), 5);
        assert_eq!(parse_max_retries(Some(" 2 ")), 2);
        assert_eq!(parse_max_retries(Some("0")), 0);
    }

    #[tokio::test]
    #[ignore = "needs a live OpenAI-compatible endpoint"]
    async fn live_chat_smoke() {
        let p = OpenAiCompatibleProvider::new("gpt-4o-mini", "https://api.openai.com/v1", "");
        let _ = p.chat("ping").await;
    }
}
