# Full-length large-scale search reproduction

This case targets the final SPLM-T5 small + paired TM head result: **957 SCOPe
queries against 430,140 Swiss-Prot targets**, with no truncation. Both the standard
(A) and low-identity (B) evaluations are reproduced. Other retrieval methods are
not re-run by this case.

**Status:** implementation and CPU contract tests (`test_large_scale.py`) are
available. Exact inference
parity and full-benchmark reproduction require the A100 validation run; they must
not be inferred from successful unit tests. Validation reports are written to the
work directory described below.

## What is fixed

- Paired backbone checkpoint `splm_tm_backbone.pth` and TM calibration head.
- Database construction, scoring and timing code. `frozen_search.py` and
  `search_cli.py` are this case's own frozen copy of the release search code;
  real-model parity with historical results remains pending. They do not import
  the public search API (`splm_tm.index`,
  `code/scripts/search_index.py`), which may change independently, nor any other
  reproduction folder. Only backbone/head loading and FASTA reading come from
  `code/splm_tm`.
- Full target and query sequences (`--max-length 0`).
- Targets partitioned at 1,024 residues; each part is encoded in original-order
  shards of at most 20,000 records, length-sorted within the shard, with maximum
  batch size 8 and an 8,192-position budget. Head batches contain at most 4,096
  vectors. Embeddings are restored to the original target order.
- Queries encoded individually, including individual head inference.
- Query and target vectors normalized again before scoring, followed by float32
  inner products, `(query @ target.T + 1) / 2` in float64, then conversion back to
  float32. This is the frozen **accuracy evaluator**; the historical speed-only
  scorer used slightly different normalization and precision. One full
  query-matrix multiplication is selected for strict comparison.
- Self-hits excluded from evaluation. B removes only positives with sequence
  identity at least 30%; negatives remain. Every query has at least ten B positives.
- AP uses score-threshold groups; P@k resolves ties by original database order.
- Full-precision per-query AP, P@1 and P@10 are compared with frozen results;
  differences cause the evaluation command to fail and are recorded explicitly.

## Files

| File | Purpose |
|---|---|
| `prepare_data.py` | Export the pack from the frozen work directory (internal; compute node) |
| `prepare_data_public.py` | Build the same pack from the published benchmark on Hugging Face |
| `run_full.py` | Re-encode everything with the released weights, score and evaluate (GPU) |
| `score_from_embeddings.py` | Score the published embeddings instead of re-encoding (CPU) |
| `evaluate.py` | Dense per-query AP / P@1 / P@10 against the frozen reference |
| `frozen_search.py` | Frozen copy of the index and scoring code this case uses |
| `check_parity.py`, `run_parity.sh` | Old-versus-new inference preflight |
| `search_cli.py` | Command-line wrapper around `frozen_search.py` |
| `reference/embedding_path.json` | Result of the verified CPU-only reproduction |
| `test_large_scale.py` | CPU checks: `python -m unittest discover -s reproduce/large_scale_search` |

## Data pack

The benchmark itself is published as
[`deepduoduo/S-PLM-TM-benchmarks`](https://huggingface.co/datasets/deepduoduo/S-PLM-TM-benchmarks) (CC BY 4.0): the
957 queries with their structures, the 430,140 target sequences and their Pfam
annotation, the 957 x 430,140 TM-align ground-truth matrix, the identity of every
positive pair, and the per-query AP / P@1 / P@10 of every method in the paper. That
is the public record of the benchmark, in its own layout. `prepare_data_public.py`
turns it into the pack this runner consumes — see *Reproducing from the published
data* below. The internal route below builds the same pack from the frozen
experiment on Hellbender, **on a compute node**:

```bash
python reproduce/large_scale_search/prepare_data.py \
  --source-work /path/to/original/stage2_work_directory \
  --reference-tag '<final embedding tag ending in __full>' --out /path/to/data
```

This copies sequences, existing TM-align labels and original A/B per-query results,
and writes query ID ordering and source paths to `provenance.json`. Structural alignment is
not repeated. The data pack contains `target.fasta`, `query.fasta`, `truth.jsonl.gz`,
`reference_per_query.tsv`, and `provenance.json`. Original data files are read only. The
output directory must be new or empty and outside the source work tree;
existing data packs are never overwritten.

## Reproducing from the published data, without a GPU

The embeddings of the published run are released, so the scoring and the
evaluation can be repeated on a CPU in minutes. Nothing is re-encoded and the
model is not needed.

```bash
hf download deepduoduo/S-PLM-TM-benchmarks --repo-type dataset \
  --include "large_scale/*" --local-dir /your/benchmarks

python reproduce/large_scale_search/prepare_data_public.py \
  --hf-dir /your/benchmarks/large_scale --out /your/data/large_scale

python reproduce/large_scale_search/score_from_embeddings.py \
  --data /your/data/large_scale \
  --embeddings /your/benchmarks/large_scale/embeddings \
  --work /your/results/large_scale

python reproduce/large_scale_search/evaluate.py \
  --data /your/data/large_scale --scores /your/results/large_scale/scores.npy \
  --out /your/results/large_scale/evaluation
```

`prepare_data_public.py` builds the same pack as `prepare_data.py` from the
published layout; the method column there is already relabelled to `S-PLM-TM`,
which is why the two scripts match different strings.
`score_from_embeddings.py` checks that the released id lists equal the pack's
order, that the vectors are L2-normalized, and then calls the same frozen
`search_embeddings` with `benchmark_tm`, so the score definition is not
reimplemented. The full query-matrix product needs about 5 GB of RAM;
`--query-block-size` lowers the peak but lets BLAS pick a different kernel.

**What to expect.** This was run and checked; see
[`reference/embedding_path.json`](reference/embedding_path.json). Every reported
metric agrees with the published value to three decimals, which is the precision
the paper reports, and P@1 is bit-identical:

| | published | reproduced on another machine | difference |
|---|---|---|---|
| MAP, setting A | 0.6380800026 | 0.6380798777 | -1.2e-07 |
| P@1, setting A | 0.8443051202 | 0.8443051202 | 0 |
| P@10, setting A | 0.8049111808 | 0.8048066876 | -1.0e-04 |
| MAP, setting B | 0.6199861084 | 0.6199853462 | -7.6e-07 |
| P@1, setting B | 0.8202716823 | 0.8202716823 | 0 |
| P@10, setting B | 0.7791013584 | 0.7789968652 | -1.0e-04 |

`evaluate.py` compares **every per-query value exactly** and therefore ends with
`NOT VERIFIED: ... per-query values differ` on this path. That is expected, not a
failure of the reproduction: the score matrix is a GEMM over float32 vectors, so
its last bits depend on the BLAS, and 1,232 of 7,656 per-query values moved, with
a median absolute difference of 4.8e-08. The P@10 difference of 1.045e-04 is
exactly 0.1/957 — one query's tenth hit changed because two scores tie
differently. Compare your `comparison.json` summary against the table above, or
against `reference/embedding_path.json`, rather than relying on the exact gate.
`run_full.py` with the released weights on the original hardware is the path that
can match exactly.

## New-versus-old inference preflight

`check_parity.py` samples actual sequences across length ranges. The original
implementation and release run in **separate Python processes** so that their
legacy module names cannot contaminate one another. Their vectors, score matrices
and full candidate rankings must match exactly. Differences from historical
cached vectors are also reported; the sample's target batch shapes differ from
the historical full run, so those diagnostics do not certify the full benchmark.

Submit `run_parity.sh` with explicit `RELEASE_ROOT`, `RESEARCH_REPO`, `SOURCE_WORK`,
`REFERENCE_TAG` and `WORK_ROOT` environment variables. It requests the original
standard hardware: 16 Xeon Gold 6338 CPU cores on an Icelake node and one A100 80GB.
Set `RUN_FULL=1` to proceed to the full run **only after preflight passes**. All
encoding and evaluation execute in the SLURM allocation, not on the login node.

## Full reproduction

Inside an appropriate compute allocation:

```bash
python reproduce/large_scale_search/run_full.py --data /path/to/data \
  --work /path/to/output --config /path/to/config.yaml \
  --checkpoint /path/to/splm_tm_backbone.pth --tm-head /path/to/paired_tm_head.pt
```

Obtain model files following `models/DOWNLOAD.md`. During internal validation the
runner can use original model paths recorded by the export; on another machine,
explicitly supply downloaded paths as above. Index construction resumes only if
the target IDs, model file paths and encoding settings match. Completed shards
recorded in `index.json` are reused after a shape check. The score matrix
sidecar stores the full target ID order, which the evaluator compares with
`target.fasta`.

The resulting `evaluation/comparison.json` includes full-precision MAP/P@1/P@10,
every changed per-query value, and `exact_match`. Rounded reference values are
MAP A **0.6381**, MAP B **0.6200**, P@1 A **0.8443**, and P@10 A **0.8049**. Merely
matching those rounded values is insufficient to mark reproduction complete.

## Runtime outputs

The full runner writes `build_timing.json` and `query_timing.json`. Encoding,
normalization, scoring, ranking and I/O have separate timings. Warm-up and model
setup are reported outside online time. The query report additionally contains
`original_profile_online_sec`: serial query encoding plus a separate speed-only
scoring profile matching the historical timing implementation. This profile uses
float32 rescaling and excludes sorting; it does not alter accuracy scores or the
per-query metric comparison. Its full-benchmark value can be compared with the
historical approximately 20.4 seconds only on the same standard hardware.
