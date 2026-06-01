# Stage 1 — cAST retrieval baseline

Target: `https://github.com/django/django` @ `1651351386ab31d8ae492c8a4922797714ca97d1`

## Run

- embedder: `voyage` / `voyage-code-3`
- chunk budget: 800 tokens (tiktoken), merge on
- corpus: 2760 files, 15198 chunks
- k: 10
- note: a general-purpose embedder is a floor; `voyage-code-3` is the code-specialized reference.

## Headline

- **recall@10 (localization, macro-avg): 0.819**
- **mean retrieved_code tokens: 3181**

recall@10 is averaged over localization tasks only. Enumeration tasks (|gold| > 10) are bounded by 10/|gold| by construction and are listed for reference, not folded into the headline.

## Per task

| task | bucket | gold | recall@10 | retrieved_tokens |
|---|---|---:|---:|---:|
| cached-property-descriptor | localization | 1 | 1.000 | 2450 |
| csrf-process-view | localization | 1 | 1.000 | 4628 |
| force-str-implementation | localization | 1 | 1.000 | 2918 |
| form-validation | localization | 2 | 1.000 | 3560 |
| m2m-changed-send-sites | localization | 6 | 0.833 | 6450 |
| paginator-page | localization | 1 | 1.000 | 3852 |
| queryset-filter-def | localization | 1 | 0.000 | 2643 |
| reverse-url-helper-def | localization | 1 | 1.000 | 3158 |
| signal-receivers-request-finished | localization | 4 | 0.000 | 3312 |
| slugify-implementation | localization | 1 | 1.000 | 2415 |
| template-render | localization | 2 | 1.000 | 1360 |
| wsgi-to-view-trace | localization | 4 | 1.000 | 2747 |
| http-response-class-tree | enumeration | 17 | 0.235 | 1861 |

