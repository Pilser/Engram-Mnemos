//! Unified LLM provider traits.
//!
//! Downstream crates depend on [`LlmProvider`] instead of a concrete
//! client so providers (OpenAI-compatible, Anthropic, local) are
//! interchangeable. [`StructuredOutput`] adds a free JSON-parsing
//! helper on top of plain chat.

use async_trait::async_trait;
use mnemos_core::Result;

/// Minimal chat interface every LLM provider must implement.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Send a single user prompt, return the assistant text.
    async fn chat(&self, prompt: &str) -> Result<String>;
    /// Send system + user messages, return the assistant text.
    async fn chat_with_system(&self, system: &str, user: &str) -> Result<String>;
}

/// Providers that can return JSON get typed deserialization for free.
#[async_trait]
pub trait StructuredOutput: LlmProvider {
    /// Chat, then parse the response body as JSON into `T`.
    async fn chat_structured<T: serde::de::DeserializeOwned + Send>(
        &self,
        prompt: &str,
    ) -> Result<T> {
        let raw = self.chat(prompt).await?;
        let value: T = serde_json::from_str(&raw)?;
        Ok(value)
    }

    /// Chat for raw JSON (providers may constrain decoding server-side;
    /// the default parses whatever `chat` returns).
    async fn chat_json(&self, prompt: &str) -> Result<serde_json::Value> {
        let raw = self.chat(prompt).await?;
        let value: serde_json::Value = serde_json::from_str(&raw)?;
        Ok(value)
    }

    /// Fetch structured output reliably: retry with a repair hint instead of
    /// trusting the model's first answer to be bare JSON.
    ///
    /// Free-form chat is unreliable for machine parsing — the model may
    /// think aloud first. Each failed parse re-asks with the previous reply
    /// quoted back plus `repair_hint`, up to `attempts` total tries.
    async fn chat_structured_with_retry<T: serde::de::DeserializeOwned + Send>(
        &self,
        prompt: &str,
        attempts: usize,
        repair_hint: &str,
    ) -> Result<T> {
        let attempts = attempts.max(1);
        let mut current = prompt.to_string();
        let mut last_err = String::new();
        for _ in 0..attempts {
            let raw = self.chat(&current).await?;
            match serde_json::from_str::<T>(extract_json(&raw)) {
                Ok(value) => return Ok(value),
                Err(e) => {
                    last_err = e.to_string();
                    current = format!(
                        "{prompt}\n\nYour previous response was not valid JSON ({last_err}). \
                         {repair_hint}\nPrevious response:\n{raw}"
                    );
                }
            }
        }
        Err(mnemos_core::MnemosError::Llm(format!(
            "structured output failed after {attempts} attempts: {last_err}"
        )))
    }
}

/// Slice the first `[`…`]` or `{`…`}` span so prose-wrapped JSON still parses.
fn extract_json(raw: &str) -> &str {
    let array = raw.find('[').zip(raw.rfind(']'));
    let object = raw.find('{').zip(raw.rfind('}'));
    match (array, object) {
        (Some((a, b)), Some((c, d))) => {
            if a < c {
                &raw[a..=b]
            } else {
                &raw[c..=d]
            }
        }
        (Some((a, b)), None) => &raw[a..=b],
        (None, Some((c, d))) => &raw[c..=d],
        (None, None) => raw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockProvider;

    #[async_trait]
    impl LlmProvider for MockProvider {
        async fn chat(&self, _prompt: &str) -> Result<String> {
            Ok(r#"{"name":"x","confidence":0.5}"#.to_owned())
        }

        async fn chat_with_system(&self, _system: &str, _user: &str) -> Result<String> {
            Ok(r#"{"name":"x","confidence":0.5}"#.to_owned())
        }
    }

    impl StructuredOutput for MockProvider {}

    #[derive(Debug, serde::Deserialize, PartialEq)]
    struct Mood {
        name: String,
        confidence: f64,
    }

    #[tokio::test]
    async fn parses_structured_output() {
        let p = MockProvider;
        let got: Mood = p.chat_structured("how do you feel?").await.unwrap();
        assert_eq!(
            got,
            Mood {
                name: "x".to_owned(),
                confidence: 0.5
            }
        );
    }
}
