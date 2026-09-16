#![recursion_limit = "256"]
//! `mnemos` binary: env-configured memory CLI plus MCP servers.
//!
//! Wiring only — all behaviour lives in the pipeline crates. The binary:
//! loads [`MnemosConfig::from_env`](mnemos_core::MnemosConfig::from_env),
//! picks chat/embedding providers from `LLM_PROVIDER` / `EMBEDDING_PROVIDER`,
//! builds one [`Storage`](mnemos_storage::Storage) per pipeline plus the
//! [`Cli`](mnemos_cli::Cli) facade, then dispatches `argv` manually (no clap).

use std::sync::Arc;

use mnemos_cli::Cli;
use mnemos_concept_extractor::LlmConceptExtractor;
use mnemos_consolidation::ConsolidationPipeline;
use mnemos_contradiction::ContradictionDetector;
use mnemos_core::{LlmConfig, MnemosConfig};
use mnemos_edge_weights::EdgeWeights;
use mnemos_embedding_local::LocalEmbeddingProvider;
use mnemos_embedding_openai::OpenAiEmbeddingProvider;
use mnemos_embedding_trait::EmbeddingProvider;
use mnemos_emotional_tagger::LlmEmotionalTagger;
use mnemos_importance_scorer::LlmImportanceScorer;
use mnemos_ingestion::IngestionPipeline;
use mnemos_llm_anthropic::AnthropicProvider;
use mnemos_llm_local::OllamaProvider;
use mnemos_llm_openai::OpenAiCompatibleProvider;
use mnemos_llm_trait::LlmProvider;
use mnemos_mcp_protocol::ProtocolTools;
use mnemos_mitosis::MitosisSplitter;
use mnemos_retrieval::RetrievalPipeline;
use mnemos_stimulation::StimulationEngine;
use mnemos_storage::Storage;

/// Load env files if present: `.env` next to binary and `.env` in current dir. All optional.
fn load_env_files() {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let _ = dotenvy::from_filename_override(dir.join(".env"));
        }
    }
    let _ = dotenvy::dotenv();
}

/// Default result count for `engram recall` when `--limit` is omitted.
pub const DEFAULT_RECALL_LIMIT: usize = 5;

/// Parsed subcommand. [`parse_args`] is the only pure, unit-testable piece
/// of the binary; everything else performs I/O.
#[derive(Debug, PartialEq)]
pub enum Command {
    /// `ingest <text...>` — store one episodic memory.
    /// `--seq <prev_id> [--seq-pos N]` chains via `TemporalSequence`.
    Ingest {
        text: String,
        prev_id: Option<u64>,
        seq_pos: Option<i64>,
    },
    /// `recall <query...> [--limit N] [--follow-seq <id> --depth N --dir up|down|both]` — recall top-`limit` memories.
    Recall {
        query: String,
        limit: usize,
        follow_seq: Option<u64>,
        depth: usize,
        dir: String,
    },
    /// `reward <score> [attributions csv | --recall-id N]` — Adam-update edge weights.
    Reward {
        score: f64,
        attributions: Vec<f64>,
        recall_id: Option<u64>,
    },
    /// `consolidate` — run one consolidation ("sleep") cycle.
    Consolidate,
    /// `setup` — create the vector index (dimension from env, see `EMBEDDING_DIM`).
    Setup,
    /// `stats` — print aggregate memory statistics as JSON.
    Stats,
    /// `status` — check embedding and LLM reachability + stats.
    Status,
    /// `train-reranker` — train the learned reranker from the feedback log.
    TrainReranker,
    /// `mcp-server` — serve the full MCP server over stdio.
    McpServer,
    /// `mcp-tools` — serve the MCP tool subset over stdio.
    McpTools,
    /// `serve` (`daemon`, `up`) — persistent daemon: HTTP (`/mcp*`, `/cli`,
    /// `/health`, `/telemetry*`) + background consolidation. Stays in terminal.
    Serve,
    /// Explicit `help` / `--help` / `-h` (agent: 5 commands only).
    Help,
    /// `help-all` / `--help-all` (operator: all commands).
    HelpAll,
    /// Anything unparseable; `message` is shown alongside the usage text.
    Invalid {
        message: String,
    },
}

/// Parse `argv` (including `argv[0]`, the program name) into a [`Command`].
///
/// Pure: no env reads, no I/O.
#[must_use]
pub fn parse_args(argv: &[String]) -> Command {
    let mut words = argv.iter().skip(1);
    let Some(command) = words.next() else {
        return Command::Invalid {
            message: "no command given".to_string(),
        };
    };
    let rest: Vec<&str> = words.map(String::as_str).collect();
    match command.as_str() {
        "ingest" => parse_ingest(&rest),
        "recall" => parse_recall(&rest),
        "reward" => parse_reward(&rest),
        "consolidate" => Command::Consolidate,
        "setup" => parse_setup(&rest),
        "stats" => Command::Stats,
        "mcp-server" => Command::McpServer,
        "mcp-tools" => Command::McpTools,
        "serve" | "daemon" | "up" => Command::Serve,
        "status" => Command::Status,
        "train-reranker" | "train_reranker" => Command::TrainReranker,
        "help" | "--help" | "-h" => Command::Help,
        "help-all" | "--help-all" => Command::HelpAll,
        other => Command::Invalid {
            message: format!("unknown command {other:?}"),
        },
    }
}

/// Parse the tail of `ingest <text...> [--seq <prev_id>] [--seq-pos N]`.
fn parse_ingest(rest: &[&str]) -> Command {
    let mut prev_id: Option<u64> = None;
    let mut seq_pos: Option<i64> = None;
    let mut text_parts: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < rest.len() {
        let w = rest[i];
        if w == "--seq" {
            i += 1;
            if let Some(v) = rest.get(i).and_then(|s| s.parse::<u64>().ok()) {
                prev_id = Some(v);
            } else {
                return Command::Invalid {
                    message: "--seq needs a number: engram ingest <text...> [--seq <prev_id>]".to_string(),
                };
            }
        } else if let Some(v) = w.strip_prefix("--seq=") {
            if let Ok(n) = v.parse::<u64>() {
                prev_id = Some(n);
            } else {
                return Command::Invalid {
                    message: "--seq needs a number: engram ingest <text...> [--seq <prev_id>]".to_string(),
                };
            }
        } else if w == "--seq-pos" {
            i += 1;
            if let Some(v) = rest.get(i).and_then(|s| s.parse::<i64>().ok()) {
                seq_pos = Some(v);
            } else {
                return Command::Invalid {
                    message: "--seq-pos needs a number: engram ingest <text...> [--seq-pos N]".to_string(),
                };
            }
        } else if let Some(v) = w.strip_prefix("--seq-pos=") {
            if let Ok(n) = v.parse::<i64>() {
                seq_pos = Some(n);
            } else {
                return Command::Invalid {
                    message: "--seq-pos needs a number: engram ingest <text...> [--seq-pos N]".to_string(),
                };
            }
        } else {
            text_parts.push(w);
        }
        i += 1;
    }
    let text = text_parts.join(" ");
    if text.trim().is_empty() {
        Command::Invalid {
            message: "ingest needs text: engram ingest <text...>".to_string(),
        }
    } else {
        Command::Ingest {
            text,
            prev_id,
            seq_pos,
        }
    }
}

/// Parse the tail of `recall <query...> [--limit N] [--follow-seq <id> --depth N --dir up|down|both]`.
fn parse_recall(rest: &[&str]) -> Command {
    let mut limit = DEFAULT_RECALL_LIMIT;
    let mut follow_seq: Option<u64> = None;
    let mut depth: usize = 10;
    let mut dir = "down".to_string();
    let mut query_parts: Vec<&str> = Vec::new();
    let mut index = 0;
    while index < rest.len() {
        let word = rest[index];
        if word == "--limit" {
            index += 1;
            let value = rest.get(index).and_then(|s| s.parse::<usize>().ok());
            let Some(value) = value else {
                return Command::Invalid {
                    message: "--limit needs a number: engram recall <query...> [--limit N]"
                        .to_string(),
                };
            };
            limit = value;
        } else if let Some(value) = word.strip_prefix("--limit=") {
            let Ok(value) = value.parse::<usize>() else {
                return Command::Invalid {
                    message: "--limit needs a number: engram recall <query...> [--limit N]"
                        .to_string(),
                };
            };
            limit = value;
        } else if word == "--follow-seq" {
            index += 1;
            let value = rest.get(index).and_then(|s| s.parse::<u64>().ok());
            let Some(value) = value else {
                return Command::Invalid {
                    message: "--follow-seq needs a number: engram recall <query...> [--follow-seq <id>]"
                        .to_string(),
                };
            };
            follow_seq = Some(value);
        } else if let Some(value) = word.strip_prefix("--follow-seq=") {
            let Ok(value) = value.parse::<u64>() else {
                return Command::Invalid {
                    message: "--follow-seq needs a number: engram recall <query...> [--follow-seq <id>]"
                        .to_string(),
                };
            };
            follow_seq = Some(value);
        } else if word == "--depth" {
            index += 1;
            let value = rest.get(index).and_then(|s| s.parse::<usize>().ok());
            let Some(value) = value else {
                return Command::Invalid {
                    message: "--depth needs a number: engram recall <query...> [--depth N]"
                        .to_string(),
                };
            };
            depth = value;
        } else if let Some(value) = word.strip_prefix("--depth=") {
            let Ok(value) = value.parse::<usize>() else {
                return Command::Invalid {
                    message: "--depth needs a number: engram recall <query...> [--depth N]"
                        .to_string(),
                };
            };
            depth = value;
        } else if word == "--dir" {
            index += 1;
            let value = rest.get(index).copied().unwrap_or("");
            if !matches!(value, "up" | "down" | "both") {
                return Command::Invalid {
                    message: "--dir must be up|down|both".to_string(),
                };
            }
            dir = value.to_string();
        } else if let Some(value) = word.strip_prefix("--dir=") {
            if !matches!(value, "up" | "down" | "both") {
                return Command::Invalid {
                    message: "--dir must be up|down|both".to_string(),
                };
            }
            dir = value.to_string();
        } else {
            query_parts.push(word);
        }
        index += 1;
    }
    let query = query_parts.join(" ");
    if query.trim().is_empty() && follow_seq.is_none() {
        Command::Invalid {
            message: "recall needs a query: engram recall <query...> [--limit N]".to_string(),
        }
    } else {
        Command::Recall {
            query,
            limit,
            follow_seq,
            depth,
            dir,
        }
    }
}

/// Parse the tail of `reward <score> [attributions csv]`.
fn parse_reward(rest: &[&str]) -> Command {
    let Some(score_word) = rest.first() else {
        return Command::Invalid {
            message: "reward needs a score: engram reward <score> [attributions csv]"
                .to_string(),
        };
    };
    let Ok(score) = score_word.parse::<f64>() else {
        return Command::Invalid {
            message: format!("reward score is not a number: {score_word:?}"),
        };
    };
    let mut attributions = Vec::new();
    let mut recall_id = None;
    if let Some(second) = rest.get(1) {
        if let Some(id_str) = second.strip_prefix("--recall-id=") {
            let Ok(id) = id_str.trim().parse::<u64>() else {
                return Command::Invalid {
                    message: format!("reward recall id is not a number: {second:?}"),
                };
            };
            recall_id = Some(id);
        } else if *second == "--recall-id" {
            let id_str = rest.get(2).copied().unwrap_or("");
            let Ok(id) = id_str.trim().parse::<u64>() else {
                return Command::Invalid {
                    message: "reward needs a number after --recall-id".to_string(),
                };
            };
            recall_id = Some(id);
        } else {
            for part in second.split(',') {
                let part = part.trim();
                if part.is_empty() {
                    continue;
                }
                let Ok(value) = part.parse::<f64>() else {
                    return Command::Invalid {
                        message: format!("reward attribution is not a number: {part:?}"),
                    };
                };
                attributions.push(value);
            }
        }
    }
    Command::Reward {
        score,
        attributions,
        recall_id,
    }
}

/// Parse `setup` (no args — dimension comes from env `EMBEDDING_DIM`).
fn parse_setup(rest: &[&str]) -> Command {
    if rest.is_empty() {
        return Command::Setup;
    }
    Command::Invalid {
        message: "setup takes no args (dimension comes from env EMBEDDING_DIM): engram setup".to_string(),
    }
}

/// Agent-focused usage: the 6 memory commands (for `help`/`--help`/`-h` and invalid).
fn usage() -> &'static str {
    "usage: engram <command> [args]\n\
     capability: episodic memory with optional sequential story chains (TemporalSequence); every recall must be rewarded via reward (use recall_id from recall); recall answers \"I don't know\" when nothing is relevant\n\
     \n\
     commands:\n\
     \x20 ingest <text...>  store one episodic memory (no sequence needed)\n\
     \x20   optionally sequential: --seq <prev_id> links this engram after <prev_id> via TemporalSequence (use previous ingest's returned id); --seq-pos N groups many engrams at same position N before next (fan-out at level)\n\
     \x20 recall <query...> [--limit N]  recall top-N memories as JSON (default 5); sequential results show [sequential: pos=N head=...]; prints \"I don't know\" when no hit clears the relevance floor (MNEMOS_RECALL_MIN_SIM, default 0.70)\n\
     \x20   optionally walk chain: --follow-seq <id> start from engram <id> (ignore query), --depth N steps (default 10), --dir up|down|both (default down) to traverse TemporalSequence\n\
     \x20 reward <score -1.0 to 1.0> [--recall-id N | attributions csv]  reward a recall based on relevancy of recalled memories so memory learns (edge weights via Adam) — must reward each recall (1.0 relevant positive, -1.0 irrelevant negative, 0 no-op)\n\
     \x20 consolidate                         run one consolidation cycle\n\
     \x20 stats                               print memory stats as JSON\n\
     \x20 status                              check embedding and LLM reachability + stats"
}

/// Operator usage: all commands (for `--help-all` / `help-all`).
fn usage_all() -> &'static str {
    "usage: engram <command> [args]\n\
     capability: episodic memory with optional sequential story chains (TemporalSequence); every recall must be rewarded via reward (use recall_id from recall); recall answers \"I don't know\" when nothing is relevant\n\
     \n\
     commands:\n\
     \x20 ingest <text...>  store one episodic memory (no sequence needed)\n\
     \x20   optionally sequential: --seq <prev_id> links after <prev_id> via TemporalSequence (use previous ingest's returned id); --seq-pos N groups many engrams at same position N before next (fan-out)\n\
     \x20 recall <query...> [--limit N]  recall top-N memories as JSON (default 5); sequential results show [sequential: pos=N head=...]; prints \"I don't know\" when no hit clears the relevance floor (MNEMOS_RECALL_MIN_SIM, default 0.70)\n\
     \x20   optionally walk chain: --follow-seq <id> start from <id> (ignore query), --depth N steps (default 10), --dir up|down|both (default down) to traverse TemporalSequence\n\
     \x20 reward <score -1.0 to 1.0> [--recall-id N | attributions csv]  reward a recall based on relevancy of recalled memories so memory learns (edge weights via Adam) — must reward each recall (1.0 relevant positive, -1.0 irrelevant negative, 0 no-op)\n\
     \x20 consolidate                         run one consolidation cycle\n\
     \x20 setup                               create Engram vector index (dim from env EMBEDDING_DIM)\n\
     \x20 stats                               print memory stats as JSON\n\
     \x20 status                              check embedding and LLM reachability + stats\n\
     \x20 train-reranker                      train the learned reranker from the feedback log (operator; then set MNEMOS_RERANKER_ALPHA>0)\n\
     \x20 mcp-server                           serve the full MCP server over stdio\n\
     \x20 mcp-tools                            serve the MCP tool subset over stdio\n\
     \x20 serve (daemon, up)                   persistent daemon: HTTP /mcp*, /cli, /health, /telemetry* + background tasks (stays in terminal)\n\
     \n\
     daemon mode:\n\
     \x20 CLI commands only hit the running daemon (POST /cli). If it is\n\
     \x20 not running, they print an error — start it with `engram serve`.\n\
     \n\
     env:\n\
     \x20 LLM_PROVIDER=openai|xai|anthropic|ollama (default openai)\n\
     \x20 EMBEDDING_PROVIDER=openai|local      (default openai)"
}

/// Build one chat provider from `LLM_PROVIDER` (default `openai`).
///
/// # Errors
///
/// Returns a message when `LLM_PROVIDER` names an unknown provider.
fn build_chat_provider(config: &LlmConfig) -> Result<Box<dyn LlmProvider>, String> {
    let name = std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "openai".to_string());
    match name.trim().to_lowercase().as_str() {
        "" | "openai" | "xai" | "grok" | "deepseek" | "deepseek-chat" | "deepseek-flash" => {
            // xAI Grok is OpenAI-compatible at https://api.x.ai/v1 — use the same path.
            // Allow `XAI_API_KEY` / `DEEPSEEK_API_KEY` to override `OPENAI_API_KEY` when set.
            let mut cfg = config.clone();
            let low = name.trim().to_lowercase();
            if low == "xai" || low == "grok" {
                if let Ok(key) = std::env::var("XAI_API_KEY") {
                    if !key.trim().is_empty() {
                        cfg.api_key = key;
                    }
                }
            } else if low.starts_with("deepseek") {
                if let Ok(key) = std::env::var("DEEPSEEK_API_KEY") {
                    if !key.trim().is_empty() {
                        cfg.api_key = key;
                    }
                }
                // Allow DEEPSEEK_BASE_URL to override OPENAI_BASE_URL when LLM_PROVIDER deepseek
                if let Ok(url) = std::env::var("DEEPSEEK_BASE_URL") {
                    if !url.trim().is_empty() {
                        cfg.base_url = url.trim().to_string();
                    }
                }
            }
            Ok(Box::new(OpenAiCompatibleProvider::from_config(&cfg)))
        }
        "anthropic" => {
            // `LlmConfig::from_env` only reads `OPENAI_API_KEY`; prefer a
            // dedicated Anthropic key when one is set.
            let mut config = config.clone();
            if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
                if !key.trim().is_empty() {
                    config.api_key = key;
                }
            }
            Ok(Box::new(AnthropicProvider::from_config(&config)))
        }
        "ollama" => Ok(Box::new(OllamaProvider::from_config(config))),
        other => Err(format!(
            "unknown LLM_PROVIDER={other:?} (expected openai|xai|anthropic|ollama)"
        )),
    }
}

/// Build one embedding provider from `EMBEDDING_PROVIDER` (default `openai`).
///
/// # Errors
///
/// Returns a message when `EMBEDDING_PROVIDER` is unknown or the local
/// model fails to download/initialise.
fn build_embedding_provider(config: &LlmConfig) -> Result<Box<dyn EmbeddingProvider>, String> {
    let name = std::env::var("EMBEDDING_PROVIDER").unwrap_or_else(|_| "openai".to_string());
    match name.trim().to_lowercase().as_str() {
        "" | "openai" => Ok(Box::new(OpenAiEmbeddingProvider::from_config(config))),
        "local" => {
            eprintln!(
                "engram: warning: local embeddings are 384-dim but the store \
                 expects 1536-dim OpenAI vectors — do NOT mix stores built with \
                 different embedding providers"
            );
            Ok(Box::new(
                LocalEmbeddingProvider::new().map_err(|error| error.to_string())?,
            ))
        }
        other => Err(format!(
            "unknown EMBEDDING_PROVIDER={other:?} (expected openai|local)"
        )),
    }
}

/// Base URL of the running daemon (`MNEMOS_MCP_HOST`/`MNEMOS_MCP_PORT`,
/// default `127.0.0.1:4545`).
fn daemon_base_url() -> String {
    let host = std::env::var("MNEMOS_MCP_HOST")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "127.0.0.1".to_string());
    let port = std::env::var("MNEMOS_MCP_PORT")
        .ok()
        .and_then(|s| s.trim().parse::<u16>().ok())
        .unwrap_or(4545);
    format!("http://{host}:{port}")
}

/// Optional bearer token for daemon requests (`MNEMOS_MCP_TOKEN`, like the server).
fn daemon_token() -> Option<String> {
    std::env::var("MNEMOS_MCP_TOKEN")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Try the running daemon for one CLI command.
///
/// Returns `Some(exit_code)` when the daemon answered (success or
/// application error), `None` when unreachable so the caller reports
/// "daemon not running". A 600ms health probe avoids hanging when nothing
/// listens. Auth mirrors the server (`Authorization: Bearer` when set).
async fn try_daemon(command: &Command) -> Option<i32> {
    let body = match command {
        Command::Ingest { text, prev_id, seq_pos } => {
            serde_json::json!({"command": "ingest", "text": text, "prev_id": prev_id, "seq_pos": seq_pos})
        }
        Command::Recall { query, limit, follow_seq, depth, dir } => {
            serde_json::json!({"command": "recall", "query": query, "limit": limit, "follow_seq": follow_seq, "depth": depth, "dir": dir})
        }
        Command::Reward { score, attributions, recall_id } => {
            serde_json::json!({"command": "reward", "score": score, "attributions": attributions, "recall_id": recall_id})
        }
        Command::Consolidate => serde_json::json!({"command": "consolidate", "aggressive": true}),
        Command::Setup => {
            serde_json::json!({"command": "setup"})
        }
        Command::Stats => serde_json::json!({"command": "stats"}),
        Command::Status => serde_json::json!({"command": "status"}),
        _ => return None,
    };
    let client = reqwest::Client::new();
    let base = daemon_base_url();
    let health = tokio::time::timeout(
        std::time::Duration::from_millis(600),
        client.get(format!("{base}/health")).send(),
    )
    .await
    .ok()?
    .ok()?;
    if !health.status().is_success() {
        return None;
    }
    let mut post = client.post(format!("{base}/cli")).json(&body);
    if let Some(token) = daemon_token() {
        post = post.bearer_auth(token);
    }
    let resp = tokio::time::timeout(std::time::Duration::from_secs(120), post.send())
        .await
        .ok()?
        .ok()?;
    let json: serde_json::Value = resp.json().await.ok()?;
    if json.get("ok").and_then(serde_json::Value::as_bool).unwrap_or(false) {
        match command {
            Command::Ingest { .. } => {
                println!("{}", json.get("data").and_then(|d| d.get("engram_id")).map_or(String::new(), ToString::to_string));
                Some(0)
            }
            Command::Reward { .. } => {
                println!("reward applied (daemon)");
                Some(0)
            }
            Command::Recall { follow_seq: Some(_), .. } => {
                // Sequential chain walk (not a relevance query).
                println!("{}", json.get("data").unwrap_or(&serde_json::Value::Null));
                Some(0)
            }
            Command::Recall { .. } => {
                let data = json.get("data").unwrap_or(&serde_json::Value::Null);
                let empty = data
                    .get("results")
                    .and_then(serde_json::Value::as_array)
                    .is_none_or(Vec::is_empty);
                if empty {
                    println!("I don't know");
                } else {
                    println!("{data}");
                }
                Some(0)
            }
            _ => {
                println!("{}", json.get("data").unwrap_or(&serde_json::Value::Null));
                Some(0)
            }
        }
    } else {
        eprintln!(
            "engram: daemon error: {}",
            json.get("error").and_then(|v| v.as_str()).unwrap_or("unknown")
        );
        Some(1)
    }
}

/// Persistent daemon: builds pipelines once, serves HTTP (`/mcp*`, `/cli`,
/// `/health`, `/telemetry*`) forever, and runs background consolidation when
/// `MNEMOS_CONSOLIDATE_INTERVAL_SECS > 0`.
async fn serve_forever(config: &MnemosConfig) -> i32 {
    // Single embedded open per process: Helix invalidates older handles when
    // the same Disk path is opened twice ("newer DB client"), so every
    // pipeline and tool shares clones of this one handle.
    let storage = match Storage::from_config(&config.storage).await {
        Ok(s) => s,
        Err(error) => {
            eprintln!("engram: error: {error}");
            return 1;
        }
    };
    let cli = match build_cli(config, storage.clone()).await {
        Ok(cli) => cli,
        Err(message) => {
            eprintln!("engram: error: {message}");
            return 1;
        }
    };
    // Background consolidation ticker ("background processing" while alive).
    let interval_secs: u64 = std::env::var("MNEMOS_CONSOLIDATE_INTERVAL_SECS")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    if interval_secs > 0 {
        let bg = Arc::clone(&cli);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
            loop {
                tick.tick().await;
                match bg.consolidate().await {
                    Ok(report) => eprintln!(
                        "engram: background consolidate pruned={} compressed={} promoted={}",
                        report.pruned, report.compressed, report.promoted
                    ),
                    Err(e) => {
                        mnemos_telemetry::global().record(
                            "mnemos-app",
                            "background_consolidate",
                            false,
                            &e.to_string(),
                        );
                    }
                }
            }
        });
        eprintln!("engram: background consolidate every {interval_secs}s");
    }
    // Protocol tools need their own provider handles (cheap HTTP clients)
    // but share the daemon's single storage handle (see above).
    let protocol = match build_protocol_tools(&cli, config, storage).await {
        Ok(p) => p,
        Err(message) => {
            eprintln!("engram: error: {message}");
            return 1;
        }
    };
    match mnemos_mcp_http::serve(protocol, cli).await {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("engram: error: serve failed: {error}");
            1
        }
    }
}

/// Assemble [`ProtocolTools`] for the daemon's `/mcp` surface.
async fn build_protocol_tools(
    cli: &Arc<Cli>,
    config: &MnemosConfig,
    storage: Storage,
) -> Result<ProtocolTools, String> {
    let llm = &config.llm;
    Ok(ProtocolTools::new(
        Arc::clone(cli),
        ContradictionDetector::new(build_chat_provider(llm)?),
        build_embedding_provider(llm)?,
        storage,
        Box::new(LlmEmotionalTagger::new(build_chat_provider(llm)?)),
    ))
}

/// Assemble all pipelines behind one [`Cli`] facade.
///
/// Takes an already-opened [`Storage`] handle and clones it for the three
/// pipelines plus the facade's own stats handle — never opens twice (see
/// `serve_forever`).
async fn build_cli(config: &MnemosConfig, storage: Storage) -> Result<Arc<Cli>, String> {

    // Each ML model needs its own `Box<dyn LlmProvider>`; the providers are
    // cheap HTTP clients, so one is built per model.
    // Detector and splitter also get their own providers (auto-wired, not opt-in).
    let detector_for_ingestion = ContradictionDetector::new(build_chat_provider(&config.llm)?);
    let splitter_for_cli = MitosisSplitter::new(build_chat_provider(&config.llm)?);
    let detector_for_cli = ContradictionDetector::new(build_chat_provider(&config.llm)?);

    let ingestion = IngestionPipeline::new(
        Box::new(LlmEmotionalTagger::new(build_chat_provider(&config.llm)?)),
        Box::new(LlmImportanceScorer::new(build_chat_provider(
            &config.llm,
        )?)),
        Box::new(LlmConceptExtractor::new(build_chat_provider(&config.llm)?)),
        build_embedding_provider(&config.llm)?,
        storage.clone(),
    )
    .with_contradiction_detector(detector_for_ingestion);
    let retrieval = RetrievalPipeline::new(
        storage.clone(),
        StimulationEngine::new(config.stimulation.clone(), EdgeWeights::defaults()),
        EdgeWeights::defaults(),
        // Second embedder instance: pipelines each own theirs. For `local`
        // this loads the on-device model twice (the Hub cache is reused, so
        // the weights download only once).
        build_embedding_provider(&config.llm)?,
    );
    let consolidation =
        ConsolidationPipeline::new(storage.clone(), config.consolidation.clone());
    Ok(Arc::new(
        Cli::new(ingestion, retrieval, consolidation, storage)
            .with_contradiction_detector(detector_for_cli)
            .with_mitosis_splitter(splitter_for_cli),
    ))
}

/// Run the binary; returns the process exit code.
async fn run(argv: Vec<String>) -> i32 {
    let command = parse_args(&argv);
    match command {
        Command::Help => {
            println!("{}", usage());
            0
        }
        Command::HelpAll => {
            println!("{}", usage_all());
            0
        }
        Command::Invalid {
            message,
        } => {
            eprintln!("engram: error: {message}\n{}", usage());
            2
        }
        Command::TrainReranker => train_reranker(),
        Command::McpServer | Command::McpTools | Command::Serve | Command::Ingest {
            ..
        } | Command::Recall {
            ..
        } | Command::Reward {
            ..
        } | Command::Setup | Command::Consolidate | Command::Stats | Command::Status => {
            dispatch(command).await
        }
    }
}

/// Build config + pipelines, then run one non-help [`Command`].
async fn dispatch(command: Command) -> i32 {
    let config = match MnemosConfig::from_env() {
        Ok(config) => config,
        Err(error) => {
            eprintln!("engram: error: failed to load config: {error}");
            return 1;
        }
    };
    eprintln!(
        "engram: storage backend=requested:{:?} effective:{:?} database={:?}",
        config.storage.backend,
        config.storage.effective_backend(),
        config.storage.database
    );
    // Persistent daemon: pipelines built once, HTTP forever + background tasks.
    if matches!(command, Command::Serve) {
        return serve_forever(&config).await;
    }
    // Thin client: CLI commands ONLY hit the running daemon (same pipelines,
    // shared learning state, background processing alive). No embedded
    // one-shot fallback — everything must be calculated by the daemon so
    // background tasks occur and telemetry records everything.
    let forwardable = matches!(
        command,
        Command::Ingest { .. }
            | Command::Recall { .. }
            | Command::Reward { .. }
            | Command::Consolidate
            | Command::Stats
            | Command::Setup
    );
    if forwardable {
        if let Some(code) = try_daemon(&command).await {
            eprintln!("engram: via daemon {}", daemon_base_url());
            return code;
        }
        eprintln!(
            "engram: error: engram daemon is not running (tried {}). Start it with `engram serve` first.",
            daemon_base_url()
        );
        mnemos_telemetry::global().record(
            "mnemos-app",
            "daemon_unreachable",
            false,
            &daemon_base_url(),
        );
        return 1;
    }
    let storage = match Storage::from_config(&config.storage).await {
        Ok(s) => s,
        Err(error) => {
            eprintln!("engram: error: {error}");
            return 1;
        }
    };
    let cli = match build_cli(&config, storage).await {
        Ok(cli) => cli,
        Err(message) => {
            eprintln!("engram: error: {message}");
            return 1;
        }
    };
    match command {
        Command::Ingest { text, prev_id, seq_pos } => {
            let out = if let Some(pid) = prev_id {
                cli.ingest_sequential(&text, Some(pid), seq_pos).await
            } else {
                cli.ingest(&text).await
            };
            match out {
                Ok(id) => {
                    println!("{id}");
                    0
                }
                Err(error) => {
                    eprintln!("engram: error: ingest failed: {error}");
                    1
                }
            }
        }
        Command::Recall { query, limit, follow_seq, depth, dir } => {
            if let Some(sid) = follow_seq {
                match cli.follow_sequence(sid, depth, &dir).await {
                    Ok(chain) => {
                        match serde_json::to_string(&serde_json::json!({"sequential_chain": chain, "start_id": sid, "depth": depth, "dir": dir})) {
                            Ok(json) => {
                                println!("{json}");
                                0
                            }
                            Err(error) => {
                                eprintln!("engram: error: failed to encode chain: {error}");
                                1
                            }
                        }
                    }
                    Err(error) => {
                        eprintln!("engram: error: follow-sequence failed: {error}");
                        1
                    }
                }
            } else {
                match cli.recall(&query, limit).await {
                    Ok(results) => {
                        if results.is_empty() {
                            println!("I don't know");
                            return 0;
                        }
                        let recall_id = cli.last_recall_id().await;
                        match serde_json::to_string(&serde_json::json!({"results": results, "recall_id": recall_id})) {
                            Ok(json) => {
                                println!("{json}");
                                0
                            }
                            Err(error) => {
                                eprintln!("engram: error: failed to encode results: {error}");
                                1
                            }
                        }
                    }
                    Err(error) => {
                        eprintln!("engram: error: recall failed: {error}");
                        1
                    }
                }
            }
        }
        Command::Reward {
            score,
            attributions,
            recall_id,
        } => match if let Some(id) = recall_id {
            cli.reward_with_id(id, score).await
        } else {
            cli.reward(&attributions, score).await
        } {
            Ok(()) => {
                println!("reward applied");
                0
            }
            Err(error) => {
                eprintln!("engram: error: reward failed: {error}");
                1
            }
        },
        Command::Consolidate => match cli.consolidate().await {
            Ok(report) => match serde_json::to_string(&report) {
                Ok(json) => {
                    println!("{json}");
                    0
                }
                Err(error) => {
                    eprintln!("engram: error: failed to encode report: {error}");
                    1
                }
            },
            Err(error) => {
                eprintln!("engram: error: consolidate failed: {error}");
                1
            }
        },
        Command::Setup => match cli.setup_vector_index(config.llm.embedding_dim).await {
            Ok(message) => {
                println!("{message}");
                0
            }
            Err(error) => {
                eprintln!("engram: error: setup failed: {error}");
                1
            }
        },
        Command::Stats => match cli.stats().await {
            Ok(stats) => match serde_json::to_string(&stats) {
                Ok(json) => {
                    println!("{json}");
                    0
                }
                Err(error) => {
                    eprintln!("engram: error: failed to encode stats: {error}");
                    1
                }
            },
            Err(error) => {
                eprintln!("engram: error: stats failed: {error}");
                1
            }
        },
        Command::Status => {
            // Storage stats
            let storage_json = match cli.stats().await {
                Ok(stats) => serde_json::json!({"ok": true, "stats": stats}),
                Err(e) => serde_json::json!({"ok": false, "error": e.to_string()}),
            };
            // Embedding reachability: try embed "ping" with 8s timeout
            let embedding_json = match build_embedding_provider(&config.llm) {
                Ok(provider) => {
                    let fut = provider.embed("ping");
                    match tokio::time::timeout(std::time::Duration::from_secs(8), fut).await {
                        Ok(Ok(v)) => serde_json::json!({"ok": true, "dim": v.len(), "provider": std::env::var("EMBEDDING_PROVIDER").unwrap_or_else(|_| "openai".to_string())}),
                        Ok(Err(e)) => serde_json::json!({"ok": false, "error": e.to_string()}),
                        Err(_) => serde_json::json!({"ok": false, "error": "timeout after 8s"}),
                    }
                }
                Err(e) => serde_json::json!({"ok": false, "error": e}),
            };
            // LLM reachability: try chat "ping" with 8s timeout, check reasoning
            let llm_json = match build_chat_provider(&config.llm) {
                Ok(provider) => {
                    let fut = provider.chat("ping");
                    match tokio::time::timeout(std::time::Duration::from_secs(8), fut).await {
                        Ok(Ok(v)) => {
                            let is_reasoning = v.contains("<think>") || v.to_lowercase().contains("reasoning") || v.len() > 500;
                            serde_json::json!({"ok": true, "provider": std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "openai".to_string()), "model": config.llm.model, "base_url": config.llm.base_url, "sample": v.chars().take(120).collect::<String>(), "reasoning_detected": is_reasoning, "note": if is_reasoning { "response looks like reasoning - may need prompt tuning for instant" } else { "instant" }})
                        }
                        Ok(Err(e)) => serde_json::json!({"ok": false, "error": e.to_string(), "provider": std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "openai".to_string())}),
                        Err(_) => serde_json::json!({"ok": false, "error": "timeout after 8s"}),
                    }
                }
                Err(e) => serde_json::json!({"ok": false, "error": e}),
            };
            let out = serde_json::json!({"storage": storage_json, "embedding": embedding_json, "llm": llm_json});
            println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
            0
        },
        // `mnemos-mcp-server` returns `mnemos_core::Result`; `mnemos-mcp-tools`
        // returns `Result<_, rmcp::RmcpError>` — both map to a message + exit 1.
        Command::McpServer => match mnemos_mcp_server::run(cli).await {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("engram: error: mcp-server failed: {error}");
                1
            }
        },
        Command::McpTools => match mnemos_mcp_tools::run(cli).await {
            Ok(()) => 0,
            Err(error) => {
                eprintln!("engram: error: mcp-tools failed: {error}");
                1
            }
        },
        Command::Help
        | Command::HelpAll
        | Command::TrainReranker
        | Command::Invalid {
            ..
        } => {
            eprintln!("engram: error: unexpected command\n{}", usage());
            2
        }
        Command::Serve => {
            eprintln!("engram: error: unexpected command\n{}", usage());
            2
        }
    }
}

/// Feature vector for one logged candidate (must match
/// [`mnemos_reranker::FEATURE_NAMES`] order).
fn reranker_features_from_log(r: &serde_json::Value, position: usize) -> Vec<f64> {
    let g = |k: &str| r.get(k).and_then(serde_json::Value::as_f64).unwrap_or(0.0);
    vec![
        g("semantic_sim"),
        g("recency_weight"),
        g("emotional_charge").abs(),
        g("importance_score"),
        g("identity_alignment"),
        g("reward_score"),
        g("reward_factor"),
        g("resonance_score"),
        position as f64,
    ]
}

/// Train the learned reranker (Option B) from the feedback log.
///
/// Labels are per-recall (one scalar reward covers the whole shown set), so
/// preference pairs are built *across* recalls: candidates from positively
/// rewarded recalls should outrank candidates from negatively rewarded ones.
fn train_reranker() -> i32 {
    use std::collections::HashMap;
    let log_path = std::env::var("MNEMOS_FEEDBACK_LOG")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "./data/helix/feedback.jsonl".to_string());
    let model_path = std::env::var("MNEMOS_RERANKER_MODEL")
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "./data/helix/reranker.json".to_string());
    let data = match std::fs::read_to_string(&log_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("engram: error: cannot read feedback log {log_path}: {e}");
            return 1;
        }
    };
    let mut candidates: HashMap<u64, Vec<Vec<f64>>> = HashMap::new();
    let mut rewards: HashMap<u64, f64> = HashMap::new();
    for line in data.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        match v.get("kind").and_then(serde_json::Value::as_str) {
            Some("recall") => {
                let Some(rid) = v.get("recall_id").and_then(serde_json::Value::as_u64) else {
                    continue;
                };
                let rows: Vec<Vec<f64>> = v
                    .get("results")
                    .and_then(serde_json::Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .enumerate()
                            .map(|(i, r)| reranker_features_from_log(r, i))
                            .collect()
                    })
                    .unwrap_or_default();
                candidates.entry(rid).or_default().extend(rows);
            }
            Some("reward") => {
                if let (Some(rid), Some(rew)) = (
                    v.get("recall_id").and_then(serde_json::Value::as_u64),
                    v.get("reward").and_then(serde_json::Value::as_f64),
                ) {
                    rewards.insert(rid, rew);
                }
            }
            _ => {}
        }
    }
    const POS: f64 = 0.2;
    const NEG: f64 = -0.2;
    let positives: Vec<&Vec<f64>> = candidates
        .iter()
        .filter(|(rid, _)| rewards.get(rid).is_some_and(|r| *r > POS))
        .flat_map(|(_, rows)| rows.iter())
        .collect();
    let negatives: Vec<&Vec<f64>> = candidates
        .iter()
        .filter(|(rid, _)| rewards.get(rid).is_some_and(|r| *r < NEG))
        .flat_map(|(_, rows)| rows.iter())
        .collect();
    if positives.is_empty() || negatives.is_empty() {
        eprintln!(
            "engram: not enough feedback yet (positive recalls={}, negative recalls={}); need both signs",
            candidates
                .keys()
                .filter(|rid| rewards.get(rid).is_some_and(|r| *r > POS))
                .count(),
            candidates
                .keys()
                .filter(|rid| rewards.get(rid).is_some_and(|r| *r < NEG))
                .count(),
        );
        return 1;
    }
    const MAX_PAIRS: usize = 200_000;
    let mut pairs: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    'outer: for p in &positives {
        for n in &negatives {
            pairs.push(((*p).clone(), (*n).clone()));
            if pairs.len() >= MAX_PAIRS {
                break 'outer;
            }
        }
    }
    let mut model = mnemos_reranker::RerankerModel::load(&model_path).unwrap_or_default();
    model.train_pairwise(&pairs, 0.05, 20);
    if let Err(e) = model.save(&model_path) {
        eprintln!("engram: error: cannot save reranker model {model_path}: {e}");
        return 1;
    }
    let out = serde_json::json!({
        "model": model_path,
        "pairs": pairs.len(),
        "positive_candidates": positives.len(),
        "negative_candidates": negatives.len(),
        "trained_samples": model.trained_samples,
        "alpha_env": "MNEMOS_RERANKER_ALPHA (default 0.0 = shadow, base CRR only)",
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap_or_default());
    0
}

#[tokio::main]
async fn main() {
    load_env_files();
    let code = run(std::env::args().collect()).await;
    if code != 0 {
        std::process::exit(code);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(words: &[&str]) -> Vec<String> {
        std::iter::once("mnemos")
            .chain(words.iter().copied())
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn ingest_joins_words() {
        assert_eq!(
            parse_args(&argv(&["ingest", "the", "sky", "is", "blue"])),
            Command::Ingest {
                text: "the sky is blue".to_string(),
                prev_id: None,
                seq_pos: None
            }
        );
    }

    #[test]
    fn ingest_without_text_is_invalid() {
        assert!(matches!(
            parse_args(&argv(&["ingest"])),
            Command::Invalid {
                ..
            }
        ));
    }

    #[test]
    fn recall_defaults_to_default_limit() {
        assert_eq!(
            parse_args(&argv(&["recall", "blue", "sky"])),
            Command::Recall {
                query: "blue sky".to_string(),
                limit: DEFAULT_RECALL_LIMIT,
                follow_seq: None,
                depth: 10,
                dir: "down".to_string()
            }
        );
    }

    #[test]
    fn recall_parses_limit_before_and_after_query() {
        assert_eq!(
            parse_args(&argv(&["recall", "--limit", "3", "blue", "sky"])),
            Command::Recall {
                query: "blue sky".to_string(),
                limit: 3,
                follow_seq: None,
                depth: 10,
                dir: "down".to_string()
            }
        );
        assert_eq!(
            parse_args(&argv(&["recall", "blue", "sky", "--limit=7"])),
            Command::Recall {
                query: "blue sky".to_string(),
                limit: 7,
                follow_seq: None,
                depth: 10,
                dir: "down".to_string()
            }
        );
    }

    #[test]
    fn recall_without_query_or_bad_limit_is_invalid() {
        assert!(matches!(
            parse_args(&argv(&["recall"])),
            Command::Invalid {
                ..
            }
        ));
        assert!(matches!(
            parse_args(&argv(&["recall", "--limit", "3"])),
            Command::Invalid {
                ..
            }
        ));
        assert!(matches!(
            parse_args(&argv(&["recall", "sky", "--limit", "many"])),
            Command::Invalid {
                ..
            }
        ));
        assert!(matches!(
            parse_args(&argv(&["recall", "sky", "--limit"])),
            Command::Invalid {
                ..
            }
        ));
    }

    #[test]
    fn reward_parses_score_and_optional_csv() {
        assert_eq!(
            parse_args(&argv(&["reward", "0.8"])),
            Command::Reward {
                score: 0.8,
                attributions: Vec::new(),
                recall_id: None,
            }
        );
        assert_eq!(
            parse_args(&argv(&["reward", "-1", "0.5, 0.25,0.125"])),
            Command::Reward {
                score: -1.0,
                attributions: vec![0.5, 0.25, 0.125],
                recall_id: None,
            }
        );
    }

    #[test]
    fn reward_parses_recall_id_forms() {
        assert_eq!(
            parse_args(&argv(&["reward", "1.0", "--recall-id", "7"])),
            Command::Reward {
                score: 1.0,
                attributions: Vec::new(),
                recall_id: Some(7),
            }
        );
        assert_eq!(
            parse_args(&argv(&["reward", "1.0", "--recall-id=9"])),
            Command::Reward {
                score: 1.0,
                attributions: Vec::new(),
                recall_id: Some(9),
            }
        );
        assert!(matches!(
            parse_args(&argv(&["reward", "1.0", "--recall-id", "nope"])),
            Command::Invalid { .. }
        ));
    }

    #[test]
    fn reward_rejects_missing_or_bad_numbers() {
        assert!(matches!(
            parse_args(&argv(&["reward"])),
            Command::Invalid {
                ..
            }
        ));
        assert!(matches!(
            parse_args(&argv(&["reward", "high"])),
            Command::Invalid {
                ..
            }
        ));
        assert!(matches!(
            parse_args(&argv(&["reward", "1.0", "0.5,nope"])),
            Command::Invalid {
                ..
            }
        ));
    }

    #[test]
    fn bare_commands_parse() {
        assert_eq!(parse_args(&argv(&["consolidate"])), Command::Consolidate);
        assert_eq!(parse_args(&argv(&["stats"])), Command::Stats);
        assert_eq!(parse_args(&argv(&["mcp-server"])), Command::McpServer);
        assert_eq!(parse_args(&argv(&["mcp-tools"])), Command::McpTools);
        assert_eq!(parse_args(&argv(&["help"])), Command::Help);
        assert_eq!(parse_args(&argv(&["--help"])), Command::Help);
        assert_eq!(parse_args(&argv(&["-h"])), Command::Help);
    }

    #[test]
    fn missing_and_unknown_commands_are_invalid() {
        assert!(matches!(
            parse_args(&argv(&[])),
            Command::Invalid {
                ..
            }
        ));
        assert!(matches!(
            parse_args(&[]),
            Command::Invalid {
                ..
            }
        ));
        assert!(matches!(
            parse_args(&argv(&["frobnicate"])),
            Command::Invalid {
                ..
            }
        ));
    }
}
