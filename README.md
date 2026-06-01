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
2. **Skeleton-first compression** *(next)* — send signatures + docstrings instead
   of full bodies when the body isn't needed, with a visible `path:start-end`
   anchor so the agent can re-read on demand.
3. **ACON-style history compression** *(planned)* — condense the agent's
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

## Layout

```
internal/retrieve/   LOCKED data contract: RetrievedChunk + Retriever
internal/meter/      token meter — tiktoken-compatible, bucketed accounting
internal/parse/      tree-sitter parsing (ported, Apache-2.0)
internal/chunk/      cAST AST chunker (Stage 1)
internal/embed/      Embedder: Voyage / OpenAI-compatible / Fake (Stage 1)
internal/index/      file watcher + SQLite vector store + retriever (Stage 1)
internal/skeleton/   skeleton-first compression (Stage 2 — planned)
internal/compress/   history compression (Stage 3 — planned)
bench/               frozen task set + gated eval + results
```

## Status

**Stage 1 (cAST retrieval) is shipped and measured** — see the baseline above.
Stage 2 (skeleton-first compression) is next: hold recall@10 while cutting the
`retrieved_code` token count, measured on the same frozen set.

## License

Apache-2.0. Portions ported from the original Cogni project (same license).
