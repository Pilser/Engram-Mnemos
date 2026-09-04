//! Importance scoring for long-term memory retention using an LLM.
//!
//! [`LlmImportanceScorer`] implements [`mnemos_ml_trait::ImportanceScorer`]
//! by asking the configured [`mnemos_llm_trait::LlmProvider`] to rate a
//! memory on a 0.0–1.0 scale, then clamping the parsed float into `[0.0, 1.0]`.

use async_trait::async_trait;
use mnemos_core::{MnemosError, Result};

/// Scores long-term retention importance (0.0 to 1.0) via an LLM provider.
pub struct LlmImportanceScorer {
    llm: Box<dyn mnemos_llm_trait::LlmProvider>,
}

impl LlmImportanceScorer {
    /// Build a scorer backed by the given LLM provider.
    ///
    /// # Errors
    ///
    /// This constructor never fails; it only stores the provider.
    #[must_use]
    pub fn new(llm: Box<dyn mnemos_llm_trait::LlmProvider>) -> Self {
        Self { llm }
    }
}

#[async_trait]
impl mnemos_ml_trait::ImportanceScorer for LlmImportanceScorer {
    /// Score one text for long-term retention importance.
    ///
    /// Sends a rating prompt to the LLM, parses the trimmed reply as `f64`,
    /// and clamps the result into `[0.0, 1.0]`.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Llm`] when the underlying provider fails or
    /// when its reply cannot be parsed as a float.
    async fn score(&self, text: &str) -> Result<f64> {
        // Prompt per doc 02: scoring dimensions spelled out so the LLM
        // weighs novelty, decisions, stakes, and specificity.
        let base = format!(
            "You are a memory importance scorer. Rate how important this \
             memory is for a personal AI to retain long-term. Consider:\n\
             - Does it contain novel information?\n\
             - Could it inform future decisions?\n\
             - Are the stakes high?\n\
             - Is it specific or generic?\n\
             Return ONLY a float from 0.0 to 1.0.\n\nMemory: {text}"
        );
        let mut prompt = base.clone();
        for attempt in 0..2 {
            match self.llm.chat(&prompt).await {
                Ok(response) => match response.trim().parse::<f64>() {
                    Ok(value) => return Ok(value.clamp(0.0, 1.0)),
                    Err(err) => {
                        mnemos_telemetry::global().record(
                            "mnemos-importance-scorer",
                            "score.parse",
                            false,
                            &format!("attempt {attempt}: {err}; reply: {response}"),
                        );
                        prompt = format!(
                            "{base}\nReply with only the number, no other words."
                        );
                    }
                },
                Err(e) => {
                    mnemos_telemetry::global().record(
                        "mnemos-importance-scorer",
                        "llm.chat",
                        false,
                        &format!("attempt {attempt}: {e}"),
                    );
                    // Retry on transient LLM errors.
                }
            }
        }
        Err(MnemosError::Llm(
            "importance scoring failed after 2 attempts".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnemos_ml_trait::ImportanceScorer;

    struct FixedProvider {
        response: String,
    }

    impl FixedProvider {
        fn boxed(response: &str) -> Box<dyn mnemos_llm_trait::LlmProvider> {
            Box::new(Self {
                response: response.to_owned(),
            })
        }
    }

    #[async_trait]
    impl mnemos_llm_trait::LlmProvider for FixedProvider {
        async fn chat(&self, _prompt: &str) -> Result<String> {
            Ok(self.response.clone())
        }

        async fn chat_with_system(&self, _system: &str, _user: &str) -> Result<String> {
            Ok(self.response.clone())
        }
    }

    #[tokio::test]
    async fn parses_valid_score() {
        let scorer = LlmImportanceScorer::new(FixedProvider::boxed("0.8"));
        assert_eq!(scorer.score("a fond memory").await.unwrap(), 0.8);
    }

    #[tokio::test]
    async fn clamps_high_score_to_one() {
        let scorer = LlmImportanceScorer::new(FixedProvider::boxed("9.9"));
        assert_eq!(scorer.score("a fond memory").await.unwrap(), 1.0);
    }

    #[tokio::test]
    async fn clamps_low_score_to_zero() {
        let scorer = LlmImportanceScorer::new(FixedProvider::boxed("-3"));
        assert_eq!(scorer.score("a fond memory").await.unwrap(), 0.0);
    }

    #[tokio::test]
    async fn unparsable_reply_returns_llm_error() {
        let scorer = LlmImportanceScorer::new(FixedProvider::boxed("nope"));
        assert!(matches!(
            scorer.score("a fond memory").await,
            Err(MnemosError::Llm(_))
        ));
    }
}
