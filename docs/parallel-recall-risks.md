# Parallel Recall & Learning — Risks, Current Safety, and Advanced Workaround

> **Decision:** Keep `Mutex<RetrievalPipeline>` (`crates/mnemos-cli/src/lib.rs:51`) as the default safe path. **Parallel-safe Recall Ledger is now always on** — in `crates/mnemos-retrieval/src/lib.rs:314` + `crates/mnemos-cli/src/lib.rs:273` note. This doc records why and how to evolve safely.

## Where learnable weights live today

| Location | File | Lifetime | Persisted? |
|---|---|---|---|
| `RetrievalPipeline::edge_weights: EdgeWeights` | `crates/mnemos-retrieval/src/lib.rs:310` | Process memory, `&mut self` via `Mutex` | **Persisted** — loads `MNEMOS_WEIGHTS_FILE` (`./data/helix/mnemos-weights.json` default) on construction and atomically `write tmp → rename` on every `reward`/`reward_with_id` |
| `StimulationEngine::weights: EdgeWeights` | `crates/mnemos-stimulation/src/lib.rs:47` | Separate clone, `crates/mnemos-app/src/main.rs:284` `::defaults()` again | **No** (still diverged — shared `Arc<RwLock` is next step, see below) |
| `RetrievalPipeline::last_attributions: Vec<f64>[8]` | `crates/mnemos-retrieval/src/lib.rs:314` | In-memory fallback for `reward([], score)` | Kept for backward compat, but rewards now prefer ledger |
| `RetrievalPipeline::ledger` | `crates/mnemos-retrieval/src/lib.rs:316` | `recall_id → attributions`, bounded at 1024, `next_recall_id` monotonic, always on | In-memory (survives parallel recalls, evicts oldest) |
| `EdgeWeights` struct itself | `crates/mnemos-edge-weights/src/lib.rs:29` | `#[derive(Serialize,Deserialize)]` | Now actually persisted when flag is on |

**During learning:** `Cli::reward` → `RetrievalPipeline::reward` → `EdgeWeights::adam_update(&mut self, attributions, score)` (`crates/mnemos-retrieval/src/lib.rs:561`) updates `m[8], v[8], t, weights[8]` in place under the exclusive `Mutex` guard. With ledger on, `reward_with_id(recall_id, score)` looks up the isolated entry, updates, then `persist_weights()` atomically writes the file. The stimulation engine's copy is **not** yet shared — reward personalizes retrieval ranking, not wave propagation (diverged at construction, see next).

**After learning:** With flag off, still only heap — exit wipes, restart = `defaults()`. With flag on, `EdgeWeights` survives via file; `ledger` is per-process (rewards must happen in same process that recalled).

## Current parallel safety

`Cli { retrieval: Mutex<RetrievalPipeline> }` serializes `recall`/`reward`. `ingest`/`stats`/`consolidate` (no `Mutex`) can run alongside a `recall` — they share only `Storage` (clone handle). `hyper`+`tower` in `crates/mnemos-mcp-http/src/lib.rs:199` handles HTTP requests in parallel, each tool `await`s the same `Arc<Cli>` — the `Mutex` is the choke point.

## Risks if we naïvely switch to `RwLock + clone`

| # | Scenario | What breaks |
|---|---|---|
| R1 | Two parallel `recall`s, then `reward([])` | `last_attributions` overwritten by second recall — reward credits wrong trace, learns wrong α |
| R2 | Parallel `recall` clones weights at `t=10`, `reward` updates to `t=11` concurrently, recall writes back stale `t=10` snapshot | Adam state rewind / lost update, ranking regresses |
| R3 | `get activation_count` → `set count+1` interleaves | One `bump_activation` lost (benign: best-effort, but undercounts) |
| R4 | Wave `recall_stimulated` reads stale α during `transfer` | One query ranks on old weights (jitter, not corruption) |

## Advanced workaround — Recall Ledger algorithm (separate, opt-in)

Goal: true parallel `recall` with correct learning and persistence, no implicit `last`.

1. **Ledger instead of `last_attributions`:** `Cli` holds `Arc<RwLock<HashMap<RecallId, Vec<f64>>>` where `RecallId = Uuid`. `recall` generates an ID, clones `EdgeWeights` under `read` lock, does vector search + scoring lock-free, then `write` inserts `id → attributions`. Returns `{id, results}` to the caller (MCP tool adds `recall_id` field; shell prints it).
2. **Explicit reward:** `reward { recall_id, score }` looks up the ledger entry (no fallback), runs `adam_update` under `write` lock, removes the entry. No `reward([])` ambiguity.
3. **Single shared `Arc<RwLock<EdgeWeights>>`** between `RetrievalPipeline` and `StimulationEngine` (or an `ArcSwap` snapshot) so wave propagation sees the same learned weights. Updates are `write`-locked, recalls clone under `read`.
4. **Persistence:** on every `adam_update`, atomic write `EdgeWeights` JSON to `MNEMOS_WEIGHTS_FILE` (default `./data/helix/mnemos-weights.json`) via `serde_json::to_string` + `rename`. On startup, `EdgeWeights::from_file_or_defaults()` loads it. Also append `RecallId → attributions → score` to a JSONL ledger for replay/diagnosis (fits `mnemos-telemetry` file sink).
5. **Queue alternative:** if HTTP bursts, replace ledger `HashMap` with `crossbeam::channel` feeding a single writer task that serializes `adam_update` — eliminates write contention entirely.

**Wired now (always on, safe by default):**

* `crates/mnemos-retrieval/src/lib.rs:316` `ledger: HashMap<u64, Vec<f64>>` + `next_recall_id: u64` (in-process, bounded 1024, `recall` inserts `id→attributions` and records `recall.ledger` telemetry; `reward_with_id` looks up, updates, persists via `persist_weights()` atomic `write tmp → rename`).
* `crates/mnemos-cli/src/lib.rs:273` `recall_protocol` always appends `[Note: consider rewarding this recall via recall_id={id} with the reward tool]` on every recall.
* Every transport passes `recall_id` identically — `crates/mnemos-mcp-protocol/src/lib.rs:338` (`:4545/mcp` protocol `recall`), `crates/mnemos-mcp-tools/src/lib.rs:153` (`:4545/mcp/tools` `mnemos_recall`), and `crates/mnemos-mcp-server` (`:4545/mcp/cli` via same `Cli`) all return `{results, recall_id}` and `reward` accepts `{recall_id, score}`.
* `crates/mnemos-retrieval/src/lib.rs:640` `reward_with_id(recall_id, score)` + `Cli::reward_with_id` (`crates/mnemos-cli/src/lib.rs:327`) and `MNEMOS_WEIGHTS_FILE` load on construction (`crates/mnemos-retrieval/src/lib.rs:336`).

This is on by default; no config needed for parallel processing. The next evolution is the shared `Arc<RwLock` weights + queue when you see real MCP contention.
