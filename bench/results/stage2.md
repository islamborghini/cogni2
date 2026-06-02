# Stage 2 — cAST + skeleton-first compression

Target: `https://github.com/django/django` @ `1651351386ab31d8ae492c8a4922797714ca97d1`

## Run

- embedder: `voyage` / `voyage-code-3`
- chunk budget: 800 tokens (tiktoken), merge on
- corpus: 2760 files, 15198 chunks
- k: 10
- full-body fraction: 0.60 (fixed)
- assembly budgets swept: [6000 5000 4000 3000 2000]; canonical run persisted at 6000.

## Result: tokens down, retrieval and syntax intact

Stage 2 changes only how retrieved chunks are *rendered*; retrieval is unchanged. This is **not** a quality claim — "no quality loss" / "same success rate" stay unproven until end-to-end (Stage 3).

- **retrieval intact**: recall@10 (localization, macro-avg) = **0.824**, identical to Stage 1 by construction — reported only as a guard that the retriever still works.
- **syntax intact**: 111/111 emitted skeletons parse as valid Python (0 failures).
- **eviction boundary**: chunks first dropped at budget **5000** — at/below it, recall@10 overstates what the agent sees (see chunks_dropped).

## Budget sweep

Lead column is **(a) skeletonization reduction** — vs the same kept chunks rendered as full bodies at the same budget. It isolates skeleton-first compression and survives the "isn't this just a smaller budget" challenge. **(b) total reduction @ budget** is vs the Stage-1 baseline and *includes* the budget cap — the total effect at that budget, not skeletonization's effect alone.

| budget | mean_tokens | (a) skeletonization reduction | (b) total reduction @ budget | chunks_dropped | tasks_with_drops |
|---:|---:|---:|---:|---:|---:|
| 6000 | 3350 | 5.7% | 2.0% | 0 | 0 |
| 5000 | 3097 | 10.7% | 7.6% | 1 | 1 |
| 4000 | 2844 | 17.6% | 14.8% | 1 | 1 |
| 3000 | 2334 | 24.2% | 27.1% | 15 | 7 |
| 2000 | 1718 | 38.0% | 44.1% | 32 | 10 |

Safety claim for Stage 2 = **parse-validity + chunks_dropped**, not recall@10. recall@10 only certifies that retrieval is unchanged.
