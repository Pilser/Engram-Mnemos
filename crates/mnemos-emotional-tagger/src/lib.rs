//! Emotional valence scoring using an [`LlmProvider`].
//!
//! [`LlmEmotionalTagger`] prompts the underlying LLM to score text from
//! -1.0 (very negative) to +1.0 (very positive), clamping out-of-range
//! replies into `[-1.0, 1.0]`.

use async_trait::async_trait;
use mnemos_core::{MnemosError, Result};

/// LLM-backed emotional valence tagger.
pub struct LlmEmotionalTagger {
    llm: Box<dyn mnemos_llm_trait::LlmProvider>,
}

impl LlmEmotionalTagger {
    /// Create a tagger from any boxed [`mnemos_llm_trait::LlmProvider`].
    #[must_use]
    pub fn new(llm: Box<dyn mnemos_llm_trait::LlmProvider>) -> Self {
        Self { llm }
    }
}

#[async_trait]
impl mnemos_ml_trait::EmotionalTagger for LlmEmotionalTagger {
    /// Score the emotional valence of `text`.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Llm`] when the LLM call fails or when the
    /// response cannot be parsed as an `f64`.
    async fn tag(&self, text: &str) -> Result<f64> {
        // Doc edge case: empty input is neutral, no LLM call needed.
        if text.trim().is_empty() {
            return Ok(0.0);
        }
        let base = format!(
            "Score the emotional valence of this text from -1.0 (very negative) to +1.0 (very positive). Return only the number.\n\nText: {text}"
        );
        // Retry once with a stricter format demand: free-form chat may
        // think aloud instead of returning a bare number.
        let mut prompt = base.clone();
        for attempt in 0..2 {
            match self.llm.chat(&prompt).await {
                Ok(response) => match response.trim().parse::<f64>() {
                    Ok(value) => return Ok(value.clamp(-1.0, 1.0)),
                    Err(e) => {
                        mnemos_telemetry::global().record(
                            "mnemos-emotional-tagger",
                            "tag.parse",
                            false,
                            &format!("attempt {attempt}: {e}; reply: {response}"),
                        );
                        prompt = format!(
                            "{base}\nReply with only the number, no other words."
                        );
                    }
                },
                Err(e) => {
                    mnemos_telemetry::global().record(
                        "mnemos-emotional-tagger",
                        "llm.chat",
                        false,
                        &format!("attempt {attempt}: {e}"),
                    );
                    // Retry on transient LLM errors.
                }
            }
        }
        Err(MnemosError::Llm(
            "emotional valence scoring failed after 2 attempts".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnemos_llm_trait::LlmProvider;

    struct MockProvider {
        reply: String,
    }

    #[async_trait]
    impl LlmProvider for MockProvider {
        async fn chat(&self, _prompt: &str) -> Result<String> {
            Ok(self.reply.clone())
        }

        async fn chat_with_system(&self, _system: &str, _user: &str) -> Result<String> {
            Ok(self.reply.clone())
        }
    }

    fn tagger_with(reply: &str) -> LlmEmotionalTagger {
        LlmEmotionalTagger::new(Box::new(MockProvider {
            reply: reply.to_owned(),
        }))
    }

    #[tokio::test]
    #[allow(clippy::float_cmp)]
    async fn parses_plain_score() {
        use mnemos_ml_trait::EmotionalTagger as _;
        let tagger = tagger_with("0.7");
        assert_eq!(tagger.tag("I am happy").await.unwrap(), 0.7);
    }

    #[tokio::test]
    #[allow(clippy::float_cmp)]
    async fn clamps_out_of_range_score() {
        use mnemos_ml_trait::EmotionalTagger as _;
        let tagger = tagger_with("2.5");
        assert_eq!(tagger.tag("overjoyed").await.unwrap(), 1.0);
    }

    #[tokio::test]
    async fn rejects_unparsable_score() {
        use mnemos_ml_trait::EmotionalTagger as _;
        let tagger = tagger_with("garbage");
        assert!(tagger.tag("meh").await.is_err());
    }

    #[tokio::test]
    #[allow(clippy::float_cmp)]
    async fn empty_text_is_neutral_without_llm() {
        use mnemos_ml_trait::EmotionalTagger as _;
        // Mock would return 0.7 if called; empty input must short-circuit.
        let tagger = tagger_with("0.7");
        assert_eq!(tagger.tag("").await.unwrap(), 0.0);
        assert_eq!(tagger.tag("   ").await.unwrap(), 0.0);
    }
}
