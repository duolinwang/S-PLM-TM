#!/bin/bash
#SBATCH --job-name splm-go
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
# GO functional consistency. If DATA_PACK has not been exported yet, SOURCE_ROOT
# (read-only splm_gvpgit) is required and the pack is exported first.
# The environment needs goatools.
#
#   sbatch --export=ALL,RELEASE_ROOT=/your/release,SOURCE_ROOT=/path/to/splm_gvpgit,DATA_PACK=/your/data/go,OUTPUT_DIR=/your/results/go \
#     /your/release/reproduce/go_functional/run_go.sh
#
# RELEASE_ROOT cannot be derived from BASH_SOURCE: sbatch runs a spooled copy.
set -euo pipefail
: "${RELEASE_ROOT:?Set RELEASE_ROOT}"
: "${DATA_PACK:?Set DATA_PACK}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
HERE="$RELEASE_ROOT/reproduce/go_functional"
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
    : "${SOURCE_ROOT:?DATA_PACK is not exported; set SOURCE_ROOT}"
    python "$HERE/prepare_go.py" --source-root "$SOURCE_ROOT" --out "$DATA_PACK"
fi
python "$HERE/run_go.py" --data "$DATA_PACK" --out "$OUTPUT_DIR" "$@"
