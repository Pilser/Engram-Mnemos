//! Configuration — env-loadable, serializable.

use serde::{Deserialize, Serialize};

/// Top-level config: `MnemosConfig::from_env()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MnemosConfig {
    #[serde(default)]
    pub storage: StorageConfig,
    #[serde(default)]
    pub llm: LlmConfig,
    #[serde(default)]
    pub stimulation: StimulationConfig,
    #[serde(default)]
    pub consolidation: ConsolidationConfig,
}

impl MnemosConfig {
    /// Load from env vars with sensible local defaults.
    /// `HELIX_URL` (default `http://localhost:6969`),
    /// `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL`.
    pub fn from_env() -> Result<Self, crate::MnemosError> {
        Ok(Self {
            storage: StorageConfig::from_env(),
            llm: LlmConfig::from_env(),
            stimulation: StimulationConfig::default(),
            consolidation: ConsolidationConfig::default(),
        })
    }
}

/// Storage backend.
///
/// Uses the `helix-db` git SDK (`branch = "main"`) with the `embedded`
/// feature: `Http` talks to a Helix server, `Embedded*` runs the engine
/// in-process via `Client::open`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// HelixDB HTTP endpoint (used by `Http` backend).
    pub url: String,
    /// Local data root for the `EmbeddedDisk` backend (default).
    pub data_root: String,
    /// Logical database name (all backends).
    pub database: String,
    /// Requested backend (see [`StorageConfig::effective_backend`]).
    #[serde(default)]
    pub backend: StorageBackend,
    /// S3 bucket for `ObjectStorage` (empty = not configured).
    #[serde(default)]
    pub object_bucket: String,
    /// S3 region for `ObjectStorage`.
    #[serde(default = "default_object_region")]
    pub object_region: String,
    /// Custom endpoint for S3-compatible stores (e.g. MinIO in Docker).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_endpoint: Option<String>,
    /// Allow plain-HTTP endpoint (local MinIO / tests only).
    #[serde(default)]
    pub object_allow_http: bool,
}

fn default_object_region() -> String {
    "us-east-1".to_string()
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl StorageConfig {
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            url: std::env::var("HELIX_URL")
                .unwrap_or_else(|_| "http://localhost:6969".to_string()),
            // Docker-friendly: mount a volume at this path to persist.
            data_root: std::env::var("MNEMOS_DATA_ROOT")
                .unwrap_or_else(|_| "./data/helix".to_string()),
            database: std::env::var("MNEMOS_DATABASE")
                .unwrap_or_else(|_| "mnemos".to_string()),
            backend: StorageBackend::from_env(),
            object_bucket: std::env::var("MNEMOS_OBJECT_BUCKET").unwrap_or_default(),
            object_region: std::env::var("MNEMOS_OBJECT_REGION")
                .unwrap_or_else(|_| "us-east-1".to_string()),
            object_endpoint: std::env::var("MNEMOS_OBJECT_ENDPOINT")
                .ok()
                .filter(|s| !s.is_empty()),
            object_allow_http: parse_bool_env("MNEMOS_OBJECT_ALLOW_HTTP", false),
        }
    }

    /// Resolve the requested backend to one that can actually open.
    ///
    /// Fallback chain: disk is the default; a disk request with an empty
    /// `data_root` (disabled/unconfigured) falls back to memory; an object
    /// request without a bucket falls back to disk, then memory. `Http`
    /// never falls back (explicit server mode).
    #[must_use]
    pub fn effective_backend(&self) -> StorageBackend {
        match self.backend {
            StorageBackend::Http | StorageBackend::EmbeddedMemory => self.backend,
            StorageBackend::EmbeddedDisk => {
                if self.data_root.trim().is_empty() {
                    StorageBackend::EmbeddedMemory
                } else {
                    StorageBackend::EmbeddedDisk
                }
            }
            StorageBackend::ObjectStorage => {
                if self.object_bucket.trim().is_empty() {
                    // Object store not configured — degrade gracefully.
                    Self {
                        backend: StorageBackend::EmbeddedDisk,
                        ..self.clone()
                    }
                    .effective_backend()
                } else {
                    StorageBackend::ObjectStorage
                }
            }
        }
    }
}

/// Parse a truthy env var (`1`/`true`/`yes`, case-insensitive).
fn parse_bool_env(name: &str, default: bool) -> bool {
    std::env::var(name).ok().map_or(default, |v| {
        matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum StorageBackend {
    /// HTTP client against a Helix server (`HELIX_URL`).
    Http,
    /// Embedded engine on local disk (`MNEMOS_DATA_ROOT`). **Default.**
    #[default]
    EmbeddedDisk,
    /// Embedded engine in-process memory (tests, fallback).
    EmbeddedMemory,
    /// Embedded engine on S3-compatible storage (needs bucket config).
    ObjectStorage,
}

impl StorageBackend {
    /// Parse `MNEMOS_BACKEND` (case-insensitive, `/`/`-`/`_` tolerant).
    ///
    /// Accepts: `disk`, `embedded`, `embedded-disk`, `memory`,
    /// `embedded-memory`, `object`, `object-storage`, `s3`, `http`, `server`.
    /// Unknown or unset values fall back to the default (`EmbeddedDisk`).
    #[must_use]
    pub fn from_env() -> Self {
        std::env::var("MNEMOS_BACKEND")
            .ok()
            .map_or(Self::default(), |v| Self::parse(&v))
    }

    /// Parse a backend name (case-insensitive, `/`/`-`/`_` tolerant).
    #[must_use]
    pub fn parse(s: &str) -> Self {
        let n = s
            .to_ascii_lowercase()
            .replace(['-', '_', '/'], "");
        match n.as_str() {
            "http" | "server" => Self::Http,
            "memory" | "embeddedmemory" | "inmemory" => Self::EmbeddedMemory,
            "object" | "objectstorage" | "s3" | "objectstore" => Self::ObjectStorage,
            _ => Self::EmbeddedDisk,
        }
    }
}

/// Unified LLM config (OpenAI-compatible chat + embeddings).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub embedding_model: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl LlmConfig {
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            base_url: std::env::var("OPENAI_BASE_URL")
                .unwrap_or_else(|_| "https://api.openai.com/v1".to_string()),
            model: std::env::var("LLM_MODEL")
                .unwrap_or_else(|_| "gpt-4o-mini".to_string()),
            embedding_model: std::env::var("EMBEDDING_MODEL")
                .unwrap_or_else(|_| "text-embedding-3-small".to_string()),
        }
    }
}

/// Spreading-activation weights (see retrieval doc edge-weight table).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StimulationConfig {
    pub alpha_recalls: f64,
    pub alpha_abstracts_to: f64,
    pub alpha_reinforces: f64,
    pub alpha_temporal: f64,
    pub alpha_contradicts: f64,
    pub alpha_defines: f64,
    pub alpha_spawned_from: f64,
    pub beta_recurrent: f64,
    pub gamma_decay: f64,
    pub tau_threshold: f64,
    pub max_iterations: usize,
    pub seed_limit: i64,
}

impl Default for StimulationConfig {
    fn default() -> Self {
        Self {
            alpha_recalls: 0.70,
            alpha_abstracts_to: 0.50,
            alpha_reinforces: 0.60,
            alpha_temporal: 0.30,
            alpha_contradicts: -0.40,
            alpha_defines: 0.20,
            alpha_spawned_from: 0.45,
            beta_recurrent: 0.35,
            gamma_decay: 0.75,
            tau_threshold: 0.15,
            max_iterations: 4,
            seed_limit: 15,
        }
    }
}

/// Thresholds for the consolidation ("sleep") cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationConfig {
    pub prune_retention: f64,
    pub prune_importance: f64,
    pub compress_retention: f64,
    pub compress_importance: f64,
    pub promote_retention: f64,
    pub promote_min_activations: i64,
    pub mitosis_min_engrams: i64,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            prune_retention: 0.05,
            prune_importance: 0.1,
            compress_retention: 0.15,
            compress_importance: 0.3,
            promote_retention: 0.8,
            promote_min_activations: 20,
            mitosis_min_engrams: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disk_cfg() -> StorageConfig {
        StorageConfig {
            url: "http://localhost:6969".to_string(),
            data_root: "./data/helix".to_string(),
            database: "mnemos".to_string(),
            backend: StorageBackend::EmbeddedDisk,
            object_bucket: String::new(),
            object_region: "us-east-1".to_string(),
            object_endpoint: None,
            object_allow_http: false,
        }
    }

    #[test]
    fn backend_defaults_to_disk() {
        assert_eq!(StorageBackend::default(), StorageBackend::EmbeddedDisk);
    }

    #[test]
    fn backend_parses_aliases() {
        assert_eq!(StorageBackend::parse("disk"), StorageBackend::EmbeddedDisk);
        assert_eq!(StorageBackend::parse("embedded"), StorageBackend::EmbeddedDisk);
        assert_eq!(StorageBackend::parse("HTTP"), StorageBackend::Http);
        assert_eq!(StorageBackend::parse("memory"), StorageBackend::EmbeddedMemory);
        assert_eq!(StorageBackend::parse("embedded-memory"), StorageBackend::EmbeddedMemory);
        assert_eq!(StorageBackend::parse("s3"), StorageBackend::ObjectStorage);
        assert_eq!(StorageBackend::parse("object-storage"), StorageBackend::ObjectStorage);
        assert_eq!(StorageBackend::parse("nonsense"), StorageBackend::EmbeddedDisk);
    }

    #[test]
    fn disk_without_root_falls_back_to_memory() {
        let cfg = StorageConfig { data_root: String::new(), ..disk_cfg() };
        assert_eq!(cfg.effective_backend(), StorageBackend::EmbeddedMemory);
    }

    #[test]
    fn object_without_bucket_falls_back_to_disk() {
        let cfg = StorageConfig {
            backend: StorageBackend::ObjectStorage,
            ..disk_cfg()
        };
        assert_eq!(cfg.effective_backend(), StorageBackend::EmbeddedDisk);
    }

    #[test]
    fn object_with_bucket_stays() {
        let cfg = StorageConfig {
            backend: StorageBackend::ObjectStorage,
            object_bucket: "helix-production".to_string(),
            ..disk_cfg()
        };
        assert_eq!(cfg.effective_backend(), StorageBackend::ObjectStorage);
    }

    #[test]
    fn http_never_falls_back() {
        let cfg = StorageConfig { backend: StorageBackend::Http, ..disk_cfg() };
        assert_eq!(cfg.effective_backend(), StorageBackend::Http);
    }
}
