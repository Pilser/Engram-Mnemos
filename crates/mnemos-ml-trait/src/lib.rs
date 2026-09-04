//! Trait definitions for all ML models.
//!
//! Concrete crates (`mnemos-emotional-tagger`, `mnemos-importance-scorer`,
//! `mnemos-concept-extractor`) implement these using an `LlmProvider`.

use async_trait::async_trait;
use mnemos_core::{ExtractedConcept, Result};

/// Scores emotional valence of text: -1.0 (very negative) to +1.0.
#[async_trait]
pub trait EmotionalTagger: Send + Sync {
    /// Tag one text.
    ///
    /// # Errors
    ///
    /// Returns [`mnemos_core::MnemosError`] when the underlying model fails.
    async fn tag(&self, text: &str) -> Result<f64>;
}

/// Scores long-term retention importance: 0.0 to 1.0.
#[async_trait]
pub trait ImportanceScorer: Send + Sync {
    /// Score one text.
    ///
    /// # Errors
    ///
    /// Returns [`mnemos_core::MnemosError`] when the underlying model fails.
    async fn score(&self, text: &str) -> Result<f64>;
}

/// Extracts key concepts from a memory.
#[async_trait]
pub trait ConceptExtractor: Send + Sync {
    /// Extract concepts with confidence scores.
    ///
    /// # Errors
    ///
    /// Returns [`mnemos_core::MnemosError`] when the underlying model fails.
    async fn extract(&self, text: &str) -> Result<Vec<ExtractedConcept>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockTagger;
    #[async_trait]
    impl EmotionalTagger for MockTagger {
        async fn tag(&self, _text: &str) -> Result<f64> {
            Ok(0.5)
        }
    }

    #[tokio::test]
    async fn mock_tagger_returns_fixed_score() {
        assert_eq!(MockTagger.tag("hello").await.unwrap(), 0.5);
    }
}
