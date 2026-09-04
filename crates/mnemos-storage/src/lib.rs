//! mnemos-storage: HelixDB client wrapper (HTTP + embedded).
//!
//! Backend is selected by [`StorageConfig::backend`]:
//! - [`StorageBackend::Http`] — `Client::new(url)`, talks to a Helix server.
//! - `EmbeddedDisk` / `EmbeddedMemory` — `Client::open(source)`, the engine
//!   runs in-process via the `embedded` feature of the `helix-db` git SDK.
//!
//! [`StorageBackend`]: mnemos_core::StorageBackend

use helix_db::{Client, HelixDbSource};
use mnemos_core::{MnemosError, StorageBackend, StorageConfig};

/// Thin wrapper over `helix_db::Client` plus the database name and URL.
#[derive(Debug, Clone)]
pub struct Storage {
    client: Client,
    database: String,
    url: String,
}

impl Storage {
    /// Build an HTTP client against the HelixDB endpoint at `url`.
    ///
    /// No network I/O — only parses the URL and constructs the client.
    pub fn new(url: &str, database: &str) -> mnemos_core::Result<Self> {
        let client = Client::new(Some(url))
            .map_err(|e| MnemosError::Storage(e.to_string()))?;
        Ok(Self {
            client,
            database: database.to_string(),
            url: url.to_string(),
        })
    }

    /// Build from [`StorageConfig`], honouring the backend selection.
    ///
    /// The backend is resolved through
    /// [`StorageConfig::effective_backend`](mnemos_core::StorageConfig::effective_backend),
    /// so unconfigured disk/object requests degrade gracefully instead of
    /// failing. Embedded variants open the engine in-process
    /// (`Client::open`) and therefore perform real I/O.
    ///
    /// # Errors
    ///
    /// Returns [`MnemosError::Storage`] when the client cannot be opened.
    pub async fn from_config(config: &StorageConfig) -> mnemos_core::Result<Self> {
        let start = std::time::Instant::now();
        let outcome: mnemos_core::Result<Self> = async {
            match config.effective_backend() {
                StorageBackend::Http => Self::new(&config.url, &config.database),
                StorageBackend::EmbeddedDisk => {
                    let source = HelixDbSource::Disk {
                        root: config.data_root.clone().into(),
                        database: config.database.clone(),
                    };
                    let client = Client::open(source)
                        .await
                        .map_err(|e| MnemosError::Storage(e.to_string()))?;
                    Ok(Self {
                        client,
                        database: config.database.clone(),
                        url: format!(
                            "embedded://disk/{}/{}",
                            config.data_root, config.database
                        ),
                    })
                }
                StorageBackend::EmbeddedMemory => {
                    let source = HelixDbSource::InMemory {
                        database: config.database.clone(),
                    };
                    let client = Client::open(source)
                        .await
                        .map_err(|e| MnemosError::Storage(e.to_string()))?;
                    Ok(Self {
                        client,
                        database: config.database.clone(),
                        url: format!("embedded://memory/{}", config.database),
                    })
                }
                StorageBackend::ObjectStorage => {
                    let source = HelixDbSource::ObjectStorage {
                        database: config.database.clone(),
                        bucket: config.object_bucket.clone(),
                        region: config.object_region.clone(),
                        endpoint: config.object_endpoint.clone(),
                        allow_http: config.object_allow_http,
                    };
                    let client = Client::open(source)
                        .await
                        .map_err(|e| MnemosError::Storage(e.to_string()))?;
                    Ok(Self {
                        client,
                        database: config.database.clone(),
                        url: format!(
                            "embedded://object/{}/{}",
                            config.object_bucket, config.database
                        ),
                    })
                }
            }
        }
        .await;
        let ms = start.elapsed().as_millis().min(u128::from(u64::MAX)) as u64;
        let detail = match &outcome {
            Ok(s) => format!("backend={:?} url={}", config.effective_backend(), s.url()),
            Err(e) => e.to_string(),
        };
        mnemos_telemetry::global().record_with_latency(
            "mnemos-storage",
            "from_config",
            outcome.is_ok(),
            &detail,
            ms,
        );
        outcome
    }

    /// Borrow the underlying HelixDB client.
    #[must_use]
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// The configured database name.
    #[must_use]
    pub fn database(&self) -> &str {
        &self.database
    }

    /// The configured endpoint URL, or an `embedded://` descriptor.
    #[must_use]
    pub fn url(&self) -> &str {
        &self.url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn from_config_builds_without_network_io() {
        let config = StorageConfig {
            backend: StorageBackend::Http,
            ..StorageConfig::default()
        };
        let storage = Storage::from_config(&config)
            .await
            .expect("from_config builds from http config");
        assert_eq!(storage.database(), config.database);
        assert_eq!(storage.url(), config.url);
    }

    #[tokio::test]
    async fn embedded_memory_opens_in_process() {
        let config = StorageConfig {
            backend: StorageBackend::EmbeddedMemory,
            ..StorageConfig::default()
        };
        let storage = Storage::from_config(&config)
            .await
            .expect("embedded in-memory opens");
        assert_eq!(storage.database(), config.database);
    }
}
