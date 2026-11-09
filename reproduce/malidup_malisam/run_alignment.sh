#!/bin/bash
#SBATCH --job-name splm-alignment
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
# MALIDUP / MALISAM residue alignment. Sequences, reference alignments and
# reference scores are all in this directory, and the alignment kernel is
# vendored, so nothing external is required. Optional: add
# CLALIGN_ROOT=/your/CLAlign to run upstream CLAlign instead of the vendored
# kernel; run.json records which one ran.
#
#   sbatch --export=ALL,RELEASE_ROOT=/your/release,OUTPUT_DIR=/your/results/alignment \
#     /your/release/reproduce/malidup_malisam/run_alignment.sh
#
# RELEASE_ROOT cannot be derived from BASH_SOURCE: sbatch runs a spooled copy.
set -euo pipefail
: "${RELEASE_ROOT:?Set RELEASE_ROOT}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
HERE="$RELEASE_ROOT/reproduce/malidup_malisam"
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
# Upstream CLAlign takes precedence when CLALIGN_ROOT points at a checkout.
if [ -n "${CLALIGN_ROOT:-}" ]; then
    export PYTHONPATH="$CLALIGN_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
fi
python "$HERE/run_alignment.py" --out "$OUTPUT_DIR" "$@"
