#!/bin/bash
#SBATCH -J splmtm_embed
#SBATCH -p gpu_requeue
#SBATCH --gres gpu:H100:1
#SBATCH -N 1 -n 8
#SBATCH --mem 64G
#SBATCH -t 02:00:00
#SBATCH -o %j_splmtm_embed.out
#SBATCH -e %j_splmtm_embed.err
#
# S-PLM-TM 编码作业（Hellbender）。
#
# 用法：
#   sbatch [--export=ALL,FASTA=...,OUT=...,MODES="raw tm"] <本脚本绝对路径>
# 或先 export 再提交：
#   FASTA=/path/my.fasta OUT=$PWD/out.npz sbatch --export=ALL <本脚本绝对路径>
#
# 卡的选择：gpu_requeue 里 A100 常年排满（曾见 244 个 A100:1 在排），H100 / L40S 好排得多。
#   短编码作业把 --gres 换成 gpu:L40S:1（48GB）通常更快拿到节点。
# ⚠️ 别用泛型 --gres gpu:1：会被调度到 A100 的 MIG 切片（9.5GB），ProtT5-XL 必 OOM。
# ⚠️ 没写 #SBATCH -A：用 `sbatch -A <your-account>` 提交，共享队列上默认账户的 FairShare 往往排不上。
# ⚠️ 本脚本如果从 Windows 传上来，先 `sed -i 's/\r$//' run_embed_hellbender.sh`。
# 不开 -e：python 的返回码要自己取并原样 exit 出去，开了 -e 就取不到
set -uo pipefail

# ---- release 根目录 ----
# ⚠️ sbatch 会把脚本**拷到 /var/spool/slurm 下执行**，所以 $BASH_SOURCE 在作业里推不出
#    release 的位置（直接用它会得到 /var/spool/slurm，python 找不到 embed.py）。
#    三级回退：显式 RELEASE_ROOT > 脚本位置（本地 bash 执行时有效）> 从提交目录向上找。
_is_release() { [ -f "$1/code/scripts/embed.py" ]; }

RELEASE_ROOT="${RELEASE_ROOT:-}"
if [ -z "$RELEASE_ROOT" ]; then
    _cand="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." 2>/dev/null && pwd || true)"
    if [ -n "$_cand" ] && _is_release "$_cand"; then RELEASE_ROOT="$_cand"; fi
fi
if [ -z "$RELEASE_ROOT" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    _d="$SLURM_SUBMIT_DIR"
    while [ "$_d" != "/" ] && [ -n "$_d" ]; do
        if _is_release "$_d"; then RELEASE_ROOT="$_d"; break; fi
        _d="$(dirname "$_d")"
    done
fi
if [ -z "$RELEASE_ROOT" ]; then
    echo "错误：推不出 S-PLM-TM 的位置。" >&2
    echo "  在提交目录不在 release 树里时，必须显式给出，例如：" >&2
    echo "    sbatch --export=ALL,RELEASE_ROOT=/path/to/S-PLM-TM[,FASTA=...,OUT=...] <本脚本>" >&2
    exit 2
fi

FASTA="${FASTA:-$RELEASE_ROOT/examples/demo.fasta}"
OUT="${OUT:-${SLURM_SUBMIT_DIR:-$PWD}/splmtm_out.npz}"
MODES="${MODES:-raw tm}"
MAX_LENGTH="${MAX_LENGTH:-}"   # 默认全长；给正整数 N 则只编码前 N 个残基
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

module load miniconda3
module load dssp 2>/dev/null || true
# ProtT5 的 tokenizer 需要 sentencepiece，请用装过它的环境
source activate "${SPLM_ENV:?Set SPLM_ENV to your conda environment}"

export TORCH_HOME="${TORCH_HOME:-$PWD/.torch_cache}"
# ProtT5-XL 冻结主干（~11GB）从这里取；没有缓存会联网下载
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# release 自包含，不要让外部 PYTHONPATH / SCOPE_EXTRA_PATH 抢先命中同名 utils / model_v6
unset PYTHONPATH SCOPE_EXTRA_PATH BACTERIA_EXTRA_PATH || true

echo "=== $(date) job ${SLURM_JOB_ID:-local} on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "release : $RELEASE_ROOT"
echo "fasta   : $FASTA"
echo "out     : $OUT"
echo "modes   : $MODES"

python "$RELEASE_ROOT/code/scripts/embed.py" \
    --fasta "$FASTA" \
    --modes $MODES \
    --out "$OUT" \
    ${MAX_LENGTH:+--max-length "$MAX_LENGTH"} \
    --batch-size "$BATCH_SIZE" \
    --max-tokens "$MAX_TOKENS" \
    $EXTRA_ARGS && rc=0 || rc=$?   # ⚠️ 必须跟在同一行或立即取：中间插任何命令（连 echo $(date)）都会把 $? 冲掉

echo "=== done rc=$rc $(date) ==="
exit $rc
