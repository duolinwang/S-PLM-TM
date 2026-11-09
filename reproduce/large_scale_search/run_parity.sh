#!/bin/bash
#SBATCH --job-name splm-release-parity
#SBATCH -p gpu
# No #SBATCH --account on purpose: submit with `sbatch -A <your-account>`.
# The directives here are the ones the published runs used, as an example;
# partition names and --constraint are site-specific, so adjust them.
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=icelake
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
set -euo pipefail

: "${RELEASE_ROOT:?Set RELEASE_ROOT to the release checkout}"
: "${RESEARCH_REPO:?Set RESEARCH_REPO to the original benchmark checkout}"
: "${SOURCE_WORK:?Set SOURCE_WORK to the frozen stage2_k1000 directory}"
: "${REFERENCE_TAG:?Set REFERENCE_TAG to the final __full embedding tag}"
: "${WORK_ROOT:?Set WORK_ROOT to a new writable validation directory}"

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
cd "$WORK_ROOT"

CASE="$RELEASE_ROOT/reproduce/large_scale_search"
echo "Job $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader
for phase in prepare original release compare; do
    python "$CASE/check_parity.py" --phase "$phase" --source-work "$SOURCE_WORK" \
        --reference-tag "$REFERENCE_TAG" --research-repo "$RESEARCH_REPO" \
        --work "$WORK_ROOT/parity_full" --max-length 0
done

# Full reconstruction is gated on exact small-sample old/release parity.
if [ "${RUN_FULL:-0}" = 1 ]; then
    python "$CASE/prepare_data.py" --source-work "$SOURCE_WORK" \
        --reference-tag "$REFERENCE_TAG" --out "$WORK_ROOT/data"
    python "$CASE/run_full.py" --data "$WORK_ROOT/data" --work "$WORK_ROOT/full"
fi
