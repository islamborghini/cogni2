# Cogni

A token-efficient context layer for coding agents. Cogni reduces the tokens an
agent (e.g. Claude Code) spends per task **without lowering how often it
succeeds** — a Pareto improvement, not a quality trade.

The headline result is a reproducible benchmark number:

> **same task success rate, N% fewer tokens.**

## What it does

Three components, each built and measured as an independent ablation:

1. **cAST retrieval** *(shipped)* — chunk the codebase along AST boundaries and
   retrieve only the chunks relevant to the task.
2. **Skeleton-first compression** *(shipped)* — send signatures + docstrings instead
   of full bodies when the body isn't needed, with a visible `path:start-end`
   anchor so the agent can re-read on demand.
3. **ACON-style history compression** *(shipped)* — condense the agent's
   accumulated history (prior tool outputs, file reads, earlier turns) between
   steps, always keeping the most recent action+observation verbatim.

The value of the project is the **per-stage number**, not one combined figure.
Each stage must produce a measured result before the next is added.

## Constraints

- Targets a **closed frontier model API** — no hidden states, logits, or
  low-level prefill. Everything runs as retrieval, static analysis, or a normal
  API call.
- "Efficiency" means tokens at a **fixed success rate**. A token reduction that
  lowers success is not a result.

## Stage 1 — cAST retrieval (shipped)

The retrieval floor every later stage is measured against. Chunks are emitted at
AST boundaries (function / method / signatures-only `class_header`), embedded, and
served by exact cosine search over a local SQLite store.

Baseline on `django/django` @ `1651351386…` (4.2.3), `voyage-code-3`, k=10,
800-token chunk budget over a 15,198-chunk corpus:

- **recall@10 (localization, macro-avg) = 0.824**
- **mean `retrieved_code` tokens = 3554**

recall@10 is averaged over localization tasks (a bounded answer set). Enumeration
tasks are reported separately — top-k retrieval can't enumerate scattered call
sites, so they're not folded into the headline. Full per-task table:
[`bench/results/stage1.md`](bench/results/stage1.md).

### Reproduce

```sh
# pin the target repo
git clone https://github.com/django/django
git -C django checkout 1651351386ab31d8ae492c8a4922797714ca97d1

# embed + measure (Voyage by default; or a local, no-quota Ollama server:
#   EMBED_PROVIDER=ollama EMBED_MODEL=mxbai-embed-large CHUNK_MAX_TOKENS=400)
VOYAGE_API_KEY=… COGNI_EVAL=1 COGNI_BENCH_REPO="$PWD/django" \
  go test -tags eval ./bench/ -run Retrieval -v -timeout 30m
```

The index is cached under `$COGNI_INDEX_DIR` (default `$TMPDIR/cogni2-index`),
keyed by repo + embedder + budget, so the corpus is embedded once and later runs
reuse the vectors.

## Stage 2 — skeleton-first compression (shipped)

Stage 2 changes only how retrieved chunks are *rendered*, not what is retrieved.
The top chunks are sent as full bodies until ~60% of an assembly budget is used;
the rest are sent as **signature + first-paragraph docstring + a body placeholder**,
each keeping its `path:start-end` anchor so the agent can re-read the body on
demand. Bodies are **omitted, not pruned** — a skeleton is never broken code.

Because retrieval is untouched, **recall is unchanged by construction** — here it is
a guard, not the result. So the Stage 2 result is framed *tokens down, retrieval and
syntax intact*; "no quality loss" / "same success rate" stay unproven until
end-to-end (Stage 3). Sweeping the assembly budget on the same corpus (recall@10
holds at **0.824**; **111/111** emitted skeletons parse as valid Python):

| budget | mean `retrieved_code` | skeletonization reduction¹ | chunks_dropped |
|---:|---:|---:|---:|
| 6000 | 3350 | 5.7% | 0 |
| 5000 | 3097 | 10.7% | 1 |
| 4000 | 2844 | 17.6% | 1 |
| 3000 | 2334 | 24.2% | 15 |
| 2000 | 1718 | 38.0% | 32 |

¹ vs. the *same kept chunks* rendered as full bodies at the same budget — isolating
skeleton-first compression from the budget cap, so it survives the "isn't this just
a smaller budget" challenge. The full report also carries total reduction vs. Stage
1, which is smaller because Stage 2 adds the re-read anchors Stage 1 didn't carry.

The safe operating point is the top of the curve: at budget 6000, skeletonization
cuts **~5.7% of retrieved-code tokens** with **zero evictions** and recall intact.
Read that number narrowly: it is 5.7% of the `retrieved_code` bucket — one of the
four the meter tracks (system / history / output are the others) — not of an agent's
total token spend, and because bodies are *omitted* the save only sticks if the agent
doesn't re-read them. Below that budget the reduction grows, but chunks begin to be
dropped at budget 5000 — the **eviction boundary** — so recall@10 stops reflecting
what the agent sees, and the safety claim becomes **parse-validity + chunks_dropped**,
not recall. Which budget to ship — and whether the re-read cost nets out — is deferred
to Stage 3, where end-to-end success rate can weigh the coverage/token tradeoff. Full
per-budget report: [`bench/results/stage2.md`](bench/results/stage2.md).

### Reproduce

```sh
# same pinned checkout as Stage 1; reuses the cached index (no re-embedding)
VOYAGE_API_KEY=… COGNI_EVAL=1 COGNI_BENCH_REPO="$PWD/django" \
  go test -tags eval ./bench/ -run Skeleton -v -timeout 30m
```

## Stage 3 — history compression (shipped)

Stage 3 condenses the agent's accumulated history (prior tool outputs, file reads,
earlier turns) between steps. It keeps the most recent action and observation verbatim,
summarizes older observations under an editable guideline, only acts once the running
history exceeds a budget, and summarizes each observation at most once.

It is measured the same way as Stages 1 and 2: on fixed inputs, varying only the
rendering. The eval replays 20 recorded agent trajectories and compares the token cost
of feeding each one back uncompressed vs. compressed. Because the action sequence is
identical, the final answer and success are **unchanged by construction**, so the token
delta is pure compression effect, with none of the run-to-run noise a live A/B test
carries. Replayed over the Django trajectories (history-budget sweep, compressed
observation size modeled deterministically):

| history budget | net cost | gross tokens | net tokens |
|---:|---:|---:|---:|
| 2000 | **19.4%** | 12.5% | -7.5% |
| 1500 | **19.1%** | 13.1% | -14.3% |
| 1000 | **18.9%** | 13.1% | -14.6% |
| 500  | **20.0%** | 13.7% | -14.4% |

Read three ways, because they say different things. **Net cost** (the headline) prices
the summarizer overhead on a cheap model and the agent tokens on the agent model, so it
is the bill you actually pay: about 20% lower. **Gross tokens** is how much smaller the
expensive agent context gets, about 13%. **Net tokens** counts the summarizer's tokens
at face value and lands near break-even, because the one-time summarization barely
amortizes on this short-horizon set. The win is in **cost**, not raw token volume.
Whether compressed context would change the agent's *decisions* is the separate live
end-to-end question, deferred (it needs a stronger or more deterministic agent than the
free-tier open model used so far). Full report:
[`bench/results/stage3.md`](bench/results/stage3.md).

### Reproduce

```sh
# replays the recorded trajectories under bench/runs/stage3/ (no API key or index needed).
# those trajectories come from a live end-to-end run and are gitignored, so the committed
# bench/results/stage3.md is the reference figure for a fresh clone.
COGNI_EVAL=1 go test -tags eval ./bench/ -run Replay -v
```

## Layout

```
internal/retrieve/   LOCKED data contract: RetrievedChunk + Retriever
internal/meter/      token meter — tiktoken-compatible, bucketed accounting
internal/parse/      tree-sitter parsing (ported, Apache-2.0)
internal/chunk/      cAST AST chunker (Stage 1)
internal/embed/      Embedder: Voyage / OpenAI-compatible / Fake (Stage 1)
internal/index/      file watcher + SQLite vector store + retriever (Stage 1)
internal/skeleton/   skeleton-first compression (Stage 2)
internal/compress/   history compression (Stage 3)
internal/agent/      multi-turn loop + OpenAI-compatible model client (Stage 3)
bench/               frozen task set + gated eval + results
```

## Status

**All three stages are shipped and measured.** Stages 1 (cAST retrieval) and 2
(skeleton-first compression) are measured offline against the frozen task set; Stage 3
(history compression) is measured by replaying recorded agent trajectories, where success
holds by construction. The remaining open thread is the live end-to-end question for
Stage 3 (whether compressed context changes the agent's decisions), which needs a stronger
or more deterministic agent than the free-tier open model used so far.

For a plain-English walkthrough of all three steps, see
[`docs/how-it-works.md`](docs/how-it-works.md).

## License

Apache-2.0. Portions ported from the original Cogni project (same license).
