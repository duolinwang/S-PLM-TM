#!/bin/bash
#SBATCH --job-name splm-scope
#SBATCH -p gpu
# No #SBATCH --account on purpose: submit with `sbatch -A <your-account>`.
# The directives here are the ones the published runs used, as an example;
# partition names and --constraint are site-specific, so adjust them.
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=icelake
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=2-00:00:00
#
# SCOPe family / superfamily retrieval. If DATA_PACK has not been exported yet,
# the frozen inputs below are required and the pack is exported first:
#   RESEARCH_REPO, SOURCE_ROOT, QUERY_PFAM, TARGET_PFAM, IDENTITY_TSV,
#   FROZEN_RAW_PICKLE, and the raw cache's MAX_LENGTH / BATCH_SIZE / MAX_TOKENS.
#
#   sbatch --export=ALL,RELEASE_ROOT=/your/release,DATA_PACK=/your/data/scope,OUTPUT_DIR=/your/results/scope \
#     /your/release/reproduce/scope_retrieval/run_scope.sh
#
# RELEASE_ROOT cannot be derived from BASH_SOURCE: sbatch runs a spooled copy.
set -euo pipefail
: "${RELEASE_ROOT:?Set RELEASE_ROOT}"
: "${DATA_PACK:?Set DATA_PACK}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
HERE="$RELEASE_ROOT/reproduce/scope_retrieval"
module load miniconda3
module load dssp
set +u
source activate "${SPLM_T5_ENV:?Set SPLM_T5_ENV to your conda environment}"
set -u
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TORCH_HOME="${TORCH_HOME:-$PWD/.torch_cache}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONIOENCODING=utf-8
unset PYTHONPATH
if [ ! -f "$DATA_PACK/manifest.json" ]; then
    python "$HERE/prepare_scope.py" --out "$DATA_PACK" \
        --research-repo "${RESEARCH_REPO:?DATA_PACK is not exported; set RESEARCH_REPO}" \
        --source-root "${SOURCE_ROOT:?set SOURCE_ROOT}" \
        --query-pfam "${QUERY_PFAM:?set QUERY_PFAM}" --target-pfam "${TARGET_PFAM:?set TARGET_PFAM}" \
        --identity-tsv "${IDENTITY_TSV:?set IDENTITY_TSV}" \
        --frozen-raw-pickle "${FROZEN_RAW_PICKLE:?set FROZEN_RAW_PICKLE}" \
        --max-length "${MAX_LENGTH:?set MAX_LENGTH}" --batch-size "${BATCH_SIZE:?set BATCH_SIZE}" \
        --max-tokens "${MAX_TOKENS:?set MAX_TOKENS}"
fi
python "$HERE/run_scope.py" --data "$DATA_PACK" --out "$OUTPUT_DIR" "$@"
