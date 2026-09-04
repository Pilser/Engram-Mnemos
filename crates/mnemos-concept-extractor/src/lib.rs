//! LLM-backed concept extraction.
//!
//! [`LlmConceptExtractor`] prompts an [`LlmProvider`](mnemos_llm_trait::LlmProvider)
//! for a JSON array of `{"name", "confidence"}` objects and normalizes the
//! result (tolerates prose wrappers, clamps confidences, drops empty names).

use async_trait::async_trait;
use mnemos_core::{ExtractedConcept, MnemosError, Result};

/// Maximum concepts kept per memory (doc edge case: trim by order returned;
/// the prompt asks the LLM to lead with the clearest concepts).
pub const MAX_CONCEPTS_PER_MEMORY: usize = 10;

/// Extracts key concepts from a memory via an LLM provider.
pub struct LlmConceptExtractor {
    llm: Box<dyn mnemos_llm_trait::LlmProvider>,
}

impl LlmConceptExtractor {
    /// Create a new extractor backed by `llm`.
    ///
    /// # Errors
    ///
    /// This constructor itself never fails; errors surface from
    /// [`ConceptExtractor::extract`](mnemos_ml_trait::ConceptExtractor::extract)
    /// when the underlying LLM call or JSON parsing fails.
    #[must_use]
    pub fn new(llm: Box<dyn mnemos_llm_trait::LlmProvider>) -> Self {
        Self { llm }
    }

    /// Extract the JSON array substring from a possibly prose-wrapped response.
    ///
    /// Accepts both a bare array and a `{"concepts": [...]}` wrapper object
    /// (the doc's example response shape).
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Llm`] when no `[`…`]` array is found or the
    /// sliced JSON fails to parse as `Vec<ExtractedConcept>`.
    fn parse_concepts(raw: &str) -> Result<Vec<ExtractedConcept>> {
        let start = raw
            .find('[')
            .ok_or_else(|| MnemosError::Llm(format!("no JSON array in LLM response: {raw}")))?;
        let end = raw
            .rfind(']')
            .ok_or_else(|| MnemosError::Llm(format!("no JSON array in LLM response: {raw}")))?;
        if end <= start {
            return Err(MnemosError::Llm(format!(
                "no JSON array in LLM response: {raw}"
            )));
        }
        let json = &raw[start..=end];
        let concepts: Vec<ExtractedConcept> = serde_json::from_str(json)
            .map_err(|e| MnemosError::Llm(format!("failed to parse concepts JSON: {e}")))?;
        Ok(concepts
            .into_iter()
            .filter(|c| !c.name.trim().is_empty())
            .map(|mut c| {
                c.confidence = c.confidence.clamp(0.0, 1.0);
                c
            })
            // Doc edge case: cap runaway lists, keep highest-confidence first.
            .take(MAX_CONCEPTS_PER_MEMORY)
            .collect())
    }
}

#[async_trait]
impl mnemos_ml_trait::ConceptExtractor for LlmConceptExtractor {
    /// Extract concepts with confidence scores from `text`.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError`] when the underlying LLM call fails or the
    /// response contains no parseable JSON array of concepts.
    async fn extract(&self, text: &str) -> Result<Vec<ExtractedConcept>> {
        // Prompt per doc 03, including the noun-phrase rules.
        let base = format!(
            "Extract key concepts, entities, and topics from this memory. \
             Return a JSON array of objects with 'name' and 'confidence' fields.\n\n\
             Rules:\n\
             - Each concept should be a specific noun phrase (2-6 words)\n\
             - Avoid overly generic terms ('stuff', 'things')\n\
             - Include technical terms, proper nouns, and domain concepts\n\
             - Confidence should reflect how clearly the concept is stated\n\n\
             Memory: {text}"
        );
        // Retry with repair: quote the bad reply back so the model converges
        // on bare JSON instead of thinking aloud.
        let mut prompt = base.clone();
        for attempt in 0..3 {
            match self.llm.chat(&prompt).await {
                Ok(response) => match Self::parse_concepts(&response) {
                    Ok(concepts) => return Ok(concepts),
                    Err(e) => {
                        mnemos_telemetry::global().record(
                            "mnemos-concept-extractor",
                            "extract.parse",
                            false,
                            &format!("attempt {attempt}: {e}"),
                        );
                        prompt = format!(
                            "{base}\n\nYour previous response was not a valid JSON array. \
                             Return ONLY the JSON array, no other text.\nPrevious response:\n{response}"
                        );
                    }
                },
                Err(e) => {
                    mnemos_telemetry::global().record(
                        "mnemos-concept-extractor",
                        "llm.chat",
                        false,
                        &format!("attempt {attempt}: {e}"),
                    );
                    // Retry on transient LLM errors.
                }
            }
        }
        Err(MnemosError::Llm(
            "concept extraction failed after 3 attempts".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnemos_llm_trait::LlmProvider;
    use mnemos_ml_trait::ConceptExtractor;

    struct MockProvider {
        response: String,
    }

    #[async_trait]
    impl LlmProvider for MockProvider {
        async fn chat(&self, _prompt: &str) -> Result<String> {
            Ok(self.response.clone())
        }

        async fn chat_with_system(&self, _system: &str, _user: &str) -> Result<String> {
            Ok(self.response.clone())
        }
    }

    fn extractor_with(response: &str) -> LlmConceptExtractor {
        LlmConceptExtractor::new(Box::new(MockProvider {
            response: response.to_owned(),
        }))
    }

    #[tokio::test]
    async fn pure_json_array_parses() {
        let ext =
            extractor_with(r#"[{"name":"rust","confidence":0.9},{"name":"memory","confidence":0.7}]"#);
        let concepts = ext.extract("some memory").await.unwrap();
        assert_eq!(concepts.len(), 2);
        assert_eq!(concepts[0].name, "rust");
        assert!((concepts[0].confidence - 0.9).abs() < 1e-9);
        assert_eq!(concepts[1].name, "memory");
        assert!((concepts[1].confidence - 0.7).abs() < 1e-9);
    }

    #[tokio::test]
    async fn prose_wrapped_json_parses() {
        let ext = extractor_with(
            r#"Here are the concepts: [{"name":"rust","confidence":0.8}] hope this helps!"#,
        );
        let concepts = ext.extract("some memory").await.unwrap();
        assert_eq!(concepts.len(), 1);
        assert_eq!(concepts[0].name, "rust");
    }

    #[tokio::test]
    async fn non_json_returns_err() {
        let ext = extractor_with("not json");
        assert!(ext.extract("some memory").await.is_err());
    }

    #[tokio::test]
    async fn confidence_clamped_to_one() {
        let ext = extractor_with(r#"[{"name":"rust","confidence":5.0}]"#);
        let concepts = ext.extract("some memory").await.unwrap();
        assert_eq!(concepts.len(), 1);
        assert!((concepts[0].confidence - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn object_wrapped_array_parses() {
        let ext = extractor_with(
            r#"{"concepts": [{"name":"Uganda ICT Hub","confidence":0.95}]}"#,
        );
        let concepts = ext.extract("some memory").await.unwrap();
        assert_eq!(concepts.len(), 1);
        assert_eq!(concepts[0].name, "Uganda ICT Hub");
    }

    #[tokio::test]
    async fn long_lists_are_capped() {
        let items: Vec<String> = (0..25)
            .map(|i| format!(r#"{{"name":"concept {i}","confidence":0.5}}"#))
            .collect();
        let ext = extractor_with(&format!("[{}]", items.join(",")));
        let concepts = ext.extract("some memory").await.unwrap();
        assert_eq!(concepts.len(), MAX_CONCEPTS_PER_MEMORY);
    }
}
