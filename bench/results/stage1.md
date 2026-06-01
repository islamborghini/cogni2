# Stage 1 — cAST retrieval baseline

Target: `https://github.com/django/django` @ `1651351386ab31d8ae492c8a4922797714ca97d1`

## Run

- embedder: `ollama` / `mxbai-embed-large`
- chunk budget: 400 tokens (tiktoken), merge on
- corpus: 2760 files, 19755 chunks
- k: 10
- note: a general-purpose embedder is a floor; `voyage-code-3` is the code-specialized reference.

## Headline

- **recall@10 (localization, macro-avg): 0.806**
- **mean retrieved_code tokens: 2038**

recall@10 is averaged over localization tasks only. Enumeration tasks (|gold| > 10) are bounded by 10/|gold| by construction and are listed for reference, not folded into the headline.

## Per task

| task | bucket | gold | recall@10 | retrieved_tokens |
|---|---|---:|---:|---:|
| cached-property-descriptor | localization | 1 | 1.000 | 2351 |
| csrf-process-view | localization | 1 | 1.000 | 2562 |
| force-str-implementation | localization | 1 | 1.000 | 2536 |
| form-validation | localization | 2 | 0.000 | 2272 |
| m2m-changed-send-sites | localization | 6 | 0.667 | 3267 |
| paginator-page | localization | 1 | 1.000 | 2188 |
| queryset-filter-def | localization | 1 | 1.000 | 1459 |
| reverse-url-helper-def | localization | 1 | 1.000 | 1831 |
| signal-receivers-request-finished | localization | 4 | 0.000 | 2260 |
| slugify-implementation | localization | 1 | 1.000 | 2202 |
| template-render | localization | 2 | 1.000 | 853 |
| wsgi-to-view-trace | localization | 4 | 1.000 | 814 |
| http-response-class-tree | enumeration | 17 | 0.294 | 1898 |
