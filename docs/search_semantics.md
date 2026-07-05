# Semantic requirement matching: research notes & design

**Problem.** Strict search mode must decide whether a product *really* satisfies every
requested characteristic. Lexical token matching fails on synonymy: «безщітковий» vs
`Brushless` in an attribute, «мат плата» vs «Материнська плата» vs `motherboard`,
«на 1200» vs `LGA 1200`/`s1200`. Hand-maintained synonym dictionaries are a dead end:
they rot, they don't scale across categories, and they never cover three languages.

## Option 1 — SPLADE / learned sparse expansion (investigated, not viable today)

SPLADE-family models produce sparse vectors over the vocabulary where the model
*expands* each text into related terms with weights — `hdmi` activates neighbouring
port/display terms, so a sparse dot-product would reward semantic overlap without any
dictionary. That would fit our existing Qdrant sparse-vector infrastructure perfectly.

What fastembed 0.8 (our embedding runtime) actually ships:

| Model | Type | Languages |
|---|---|---|
| `prithivida/Splade_PP_en_v1` | SPLADE++ | **English only** |
| `Qdrant/bm42-all-minilm-l6-v2-attentions` | attention-weighted BM42 | English (all-MiniLM) |
| `Qdrant/bm25` | classic BM25 | language-configurable stemmer (what we use) |
| `Qdrant/minicoil-v1` | contextual word weights | English |

**Verdict:** there is no multilingual SPLADE-class checkpoint in fastembed — every
learned-sparse option is English-trained, and our catalog/queries are uk/ru/en mixed.
Swapping the sparse side to `Splade_PP_en_v1` would *break* Cyrillic lexical matching
(BM25 currently carries it) while only helping the Latin fraction of the text.

**Escalation path (when justified):** BGE-M3 produces multilingual dense + sparse
(lexical-weight) representations in one model and would replace both of our vectors.
It is not in fastembed; it requires the torch/sentence-transformers runtime (already an
optional dependency group for the reranker), a heavier container, and a **full reindex**
(guarded by `service_meta`). Worth revisiting if strict-mode quality on long-tail
attribute synonymy becomes the bottleneck.

## Option 2 — LLM query understanding via local Ollama (chosen)

The host already runs Ollama (`gemma4:e4b`, RTX 3080 Ti). Instead of expanding the
*index*, we expand the *query* — once per unique query, dictionary-free. **We tokenize,
the LLM only enriches**: the significant tokens are computed by the service and sent to
the model, which returns variants per token:

```
Query:  "безщітковий шуруповерт"
Tokens: ["безщітковий", "шуруповерт"]
  → LLM →
{"безщітковий": ["brushless", "бесщеточный", "безщеточний"],
 "шуруповерт":  ["шуруповерт", "screwdriver", "drill driver"]}
```

Each of *our* tokens becomes one requirement (token itself always the first variant).
Coverage then checks each requirement against the product's own fields: a requirement
is satisfied when **any** of its variants matches lexically (digit-boundary rules for
numbers, prefix rules for words — see `app/services/coverage.py`). `missing_terms`
reports requirement names, so the client sees *what* is missing.

Why this shape:

- **The LLM is the synonymizer** — translations, abbreviations and format variants are
  generated per query, never maintained by hand.
- **The LLM cannot drop or invent constraints — by construction.** The first design
  asked the model to segment the query itself; measured on `gemma4:e4b` it silently
  omitted «безщітковий» from "безщітковий шуруповерт makita", which would make strict
  mode over-promise. With token expansion, requirement *identity* is deterministic:
  tokens missing from the answer degrade to single-variant requirements, extra keys in
  the answer are ignored.
- **One call per unique query, strict mode only.** Results are LRU-cached (1024
  entries, keyed by token list); relaxed mode never pays LLM latency and keeps
  heuristic per-token coverage.
- **Graceful degradation is mandatory.** Timeout, connection error, or unusable JSON →
  fall back to per-token requirements; search never depends on Ollama being up.
  Outcomes are visible in Prometheus (`query_understanding_total{outcome}`).
- **Guardrails on untrusted LLM output**: JSON validated, variant counts and lengths
  capped, everything lowercased; the prompt forbids broader-term variants (a residual
  risk that a broad variant like "port" sneaks in remains — it can only ever make a
  *single* requirement easier to satisfy, never fabricate a whole product match).
- **Measured latency** (RTX 3080 Ti under WSL2, warm model): 6–16 s per uncached query
  (GPU paravirtualization overhead); budget 20 s, cold load (~45 s) is paid once thanks
  to `keep_alive=30m` + a warmup call at service start.
- **Measured model limits** (`gemma4:e4b`): uk→en translation of rarer technical terms
  can miss — «безщітковий» came back as "cordless"/"akumulyatornyi", not "brushless",
  while «мат»/«плата» → "motherboard" is reliable. This bounds query-time
  synonymization quality and is the strongest argument for **ingest-time enrichment**:
  with the product card in context (attribute value `Brushless`), generating uk/ru
  aliases is a far easier task than blind query-side translation. See
  `pipeline_assessment.md`.

Configuration: `QUERY_LLM_ENABLED`, `OLLAMA_URL`, `QUERY_LLM_MODEL`,
`QUERY_LLM_TIMEOUT_S`. In Docker the API reaches the host's Ollama via
`host.docker.internal` (mapped with `host-gateway`).

## Rejected alternatives

- **Per-attribute dense similarity** (embed every candidate's attributes at query time
  with the existing e5 model): ~50 hits × ~10 attribute strings per query on CPU —
  seconds of latency; would need index-time attribute embeddings to be viable.
- **Static synonym dictionaries / rule routers:** explicitly out — unmaintainable,
  three languages, unbounded attribute vocabulary.
