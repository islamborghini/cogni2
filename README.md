# Cogni

A token-efficient context layer for coding agents. Cogni reduces the tokens an
agent (e.g. Claude Code) spends per task **without lowering how often it
succeeds** — a Pareto improvement, not a quality trade.

The headline result is a reproducible benchmark number:

> **same task success rate, N% fewer tokens.**

## What it does

Three components, each built and measured as an independent ablation:

1. **cAST retrieval** — chunk the codebase along AST boundaries and retrieve only
   the chunks relevant to the task.
2. **Skeleton-first compression** — send signatures + docstrings instead of full
   bodies when the body isn't needed, with a visible `path:start-end` anchor so
   the agent can re-read on demand.
3. **ACON-style history compression** — condense the agent's accumulated history
   (prior tool outputs, file reads, earlier turns) between steps, always keeping
   the most recent action+observation verbatim.

The value of the project is the **per-stage number**, not one combined figure.
Each stage must produce a measured result before the next is added.

## Constraints

- Targets a **closed frontier model API** — no hidden states, logits, or
  low-level prefill. Everything runs as retrieval, static analysis, or a normal
  API call.
- "Efficiency" means tokens at a **fixed success rate**. A token reduction that
  lowers success is not a result.

## Layout

```
internal/retrieve/   LOCKED data contract: RetrievedChunk + Retriever
internal/meter/      token meter — tiktoken-compatible, bucketed accounting
internal/parse/      tree-sitter parsing (ported, Apache-2.0)
internal/index/      file watcher + (Stage 1) vector store
internal/chunk/      AST chunker (Stage 1)
internal/embed/      Embedder interface (Stage 1)
internal/skeleton/   skeleton-first compression (Stage 2)
internal/compress/   history compression (Stage 3)
bench/               frozen task set + per-stage evaluations & results
```

## Status

Foundation scaffolded. Stage 1 (cAST retrieval) is the next deliverable; its
`recall@10` and mean `retrieved_code` tokens become the baseline every later
stage is compared against.

## License

Apache-2.0. Portions ported from the original Cogni project (same license).
