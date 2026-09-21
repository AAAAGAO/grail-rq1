# GRAIL RQ1: current Full method

Minimal self-contained distribution of the evaluated V12 Full retrieval method
and nine frozen datasets (30 queries each). No local experiment results, model
response caches, credentials, or comparison baselines are included.

## Setup

Use Python 3.11 and install the dependencies:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Set `DASHSCOPE_API_KEY` in your environment. Never commit a key. The default
endpoint is the Alibaba Cloud Tokyo deployment used in the experiments; your
account must have access to that deployment. Override it with `--endpoint` or
`ALIYUN_ENDPOINT` when necessary, and report the change in comparisons.

```sh
python run_rq1.py --jobs 3
python run_rq1.py --model qwen3.8-flash --output outputs/qwen --jobs 3
python run_rq1.py --datasets data --output outputs/data
```

These commands make paid model calls. Every query is checkpointed in the output
directory. Use separate output directories for different models/configurations.
Do not resume a completed partial run merely to select better answers: successful
queries are reused, but failed queries are attempted again on resume.

## Frozen protocol

- Initial BM25 pair pool: 30; V12 adaptive exploration and online candidate review.
- Maximum six tool actions, three expansions, four frontier APIs per expansion.
- Read six KUs at a time; retain up to 30 candidate pairs.
- All structural and semantic API edges enabled.
- Terminal independent scoring in batches of six; tied scores keep controller order.
- Default model: `deepseek-v4.1-flash`; no model fallback.
- Labels are used after retrieval, not supplied to prompts.

`rq1_summary.json` divides by all 30 queries, treating failures as zero. Per-run
`summary.json` is the original runner's successful-query-only diagnostic and
must not be confused with the full-denominator RQ1 summary. MRR is truncated at
15 and uses only the first positive pair. A positive pair has a query-gold API
and `relevant_groundtruth == 1`; this is not independent query-specific KU labeling.

## Contents and provenance

`data/graphs/<dataset>/` contains knowledge pairs, API nodes, API edges and edge
evidence. `data/queries/` contains the evaluation queries and gold API labels.
`data/RE_reference/` contains required occurrence-specific reference corrections.
These are frozen retrieval inputs, even though the original local project stored
them beneath an experiment-output directory. Input SHA256 checksums are in
`INPUT_SHA256.json`.

Implementation modules are copied unchanged from the evaluated project to avoid
changing retrieval behavior while packaging. Some supporting modules retain old
command-line functions; `run_rq1.py` is the supported current-protocol entry point.
The `baselines/` directory contains an imported text helper, not RQ1 competitors.
Upstream graph construction and identification training are outside this package.

The corpus contains third-party documentation and knowledge-unit text. This
private research snapshot does not grant redistribution rights to those materials.
Keep original source fields and check upstream licenses before making it public.
No new blanket license for third-party data is asserted here.
