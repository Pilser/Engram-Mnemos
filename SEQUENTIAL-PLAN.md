# Sequential Memory Plan — TemporalSequence chain

> Additive, CI-only build. Local: `cargo check` only.

## Goal
Enable sequential/story memories: ingest can chain engrams via `TemporalSequence` Engram→Engram, return chain ids, recall annotates sequential position + allows walking.

Current state (verified):
- `mnemos-ingestion/src/lib.rs:199:200` only `Recalls`+`AbstractsTo`, no `TemporalSequence`/`Reinforces` creator (5 hits docs/retrieval only)
- `retrieval/src/lib.rs:602:605` wave already queries `TemporalSequence` via `stimulation/src/lib.rs:31:43` `neighbors_query` but returns empty
- `ResonanceResult` `core/src/types.rs:119:128` has no sequential fields

## Design (safe additive)

### 1. Ingestion `crates/mnemos-ingestion/src/lib.rs`
- New query `connect_temporal_sequence_edge(from_id: i64, to_id: i64)` `g().n(from).add_e("TemporalSequence", g().n(to), vec![])` like `contradiction/src/lib.rs:72:90`
- New API `ingest_sequential(&self, text, kind, importance, prev_id: Option<EngramId>, seq_pos: Option<i64>) -> Result<EngramId>` : calls `ingest_inner` then if `Some(prev)` `connect_temporal_sequence_edge(prev,new)` (id validated via `i64::try_from`). `seq_pos` stored as edge property `("pos", seq_pos)` if provided, else omitted. No change to `ingest`/`ingest_with_importance` paths.
- Optional grouping: same `prev_id` + same `pos` = fan-out at same level (multiple engrams before next `pos`) — DAG not just linear.

### 2. Core `crates/mnemos-core/src/types.rs`
- Optional `SequenceInfo { sequential: bool, position: u64, seq_head: EngramId, prev: Option<EngramId>, next: Vec<EngramId>, depth: usize }` or keep in retrieval formatting layer. No schema migration — edge-only.

### 3. Retrieval `crates/mnemos-retrieval/src/lib.rs`
- Add `#[query] get_temporal_prev/next(engram_id)` `in_("TemporalSequence")` / `out("TemporalSequence")` and `get_sequence_head` walk-back via `neighbors` loop.
- `cli/src/lib.rs:326:337` `recall_protocol_inner` enrich: for each `ResonanceResult` if `in/out` non-empty append `[sequential: pos=N head=...]` (position = hops from head via `neighbors` traversal, head = node with `in==0`).
- New helper `recall_follow_sequence(seq_id, depth, dir: up|down|both)` using `StimulationEngine::neighbors` `stimulation/src/lib.rs:129:147` loop with `max_iterations=depth`.

### 4. CLI `crates/mnemos-cli/src/lib.rs`
- `Cli::ingest_sequential(text, prev_id, seq_pos)` and `Cli::follow_sequence(id, depth, dir)`
- Keep `Cli::ingest/recall` unchanged.

### 5. App `crates/mnemos-app/src/main.rs`
- `Command::Ingest { text, prev_id, seq_pos }` parse `ingest <text...> [--seq <prev_id>] [--seq-pos <N>]` `parse_args:99:124` (optional, backward-compat)
- `Command::Recall { query, limit, follow_seq, depth, dir }` parse `recall <query> [--limit N] [--follow-seq <id> --depth N --dir up|down|both]`
- `usage()` stays 5 agent commands, `usage_all()` adds `--seq` notes.
- `try_daemon:352:410` body includes `prev_id, seq_pos, follow_seq`.

### 6. MCP `crates/mnemos-mcp-server/src/lib.rs` + `crates/mnemos-mcp-tools/src/lib.rs` + `crates/mnemos-mcp-http/src/lib.rs:364:418`
- `MnemosCliParams` add `prev_id, seq_pos, follow_seq, seq_depth, seq_dir` optional.
- `engram_cli` description update, `topic_help` `ingest`/`recall` examples.
- `engram_ingest` tool `text*, prev_id?, seq_pos?` ; `engram_recall` add `follow_seq?, depth?, dir?`.
- `dispatch_cli_rpc` `ingest` branch: if `prev_id` call `ingest_sequential` else `ingest`.

### 7. Build & Verify
- Local: `cargo check --workspace --all-targets` only
- Push → CI `ci` (check/clippy/test/build release) ` .github/workflows/ci.yml:49:68` uploads `target/release/engram`
- After green: `gh api .../artifacts/.../zip` → `unzip` → test `engram --help` (5), `engram --help-all` (full), sequential: `id=$(./engram ingest "ch1"); ./engram ingest "ch2" --seq $id` → recall shows `[sequential]`.

## Non-goals / Safety
- Do not reuse `Concept.source_count`/`mitosis` HDBSCAN for order (similarity destroys sequence `mitosis/src/lib.rs:63:75`)
- Do not change `consolidation/src/lib.rs:138:203` decay logic or `EdgeWeights` `temporal_seq:0.30` `edge-weights/src/lib.rs:37` — weight already lowest, wave remains weighted hop not re-sort.
- Edges property-less today `vec![]`; adding optional `pos` is additive, old edges still valid.

## Implementation Order
1. ingestion edge + API
2. retrieval annotation
3. cli + app + mcp surfaces
4. cargo check locally, commit/push, CI build, download to `deploy/engram` + `FPE/AI-plan/deployment/engram/engram`
