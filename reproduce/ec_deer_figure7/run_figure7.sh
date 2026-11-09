#!/bin/bash
#SBATCH -J ec_fig7
#SBATCH -p gpu_requeue
#SBATCH --gres gpu:L40S:1
#SBATCH -c 8
#SBATCH --mem 64G
#SBATCH -t 02:00:00
#SBATCH -o %j_ec_fig7.out
#SBATCH -e %j_ec_fig7.err
#
# 复现 EC head 在 DEER Figure 7 上的发布数字。**完全自包含**：序列、真值、打分代码
# 都在本目录，不依赖任何研究仓库。
#
#   RELEASE=<release 根目录的绝对路径>
#   mkdir -p ~/ec_fig7 && cd ~/ec_fig7
#   sbatch --export=ALL,RELEASE_ROOT=$RELEASE $RELEASE/reproduce/ec_deer_figure7/run_figure7.sh
#
# 三步：
#   A  data/{templates,database}.fasta --embed.py--> emb_h（--modes tm，**max-length 1024**）
#   B  score_figure7.py：套 EC head -> 检索 -> 按 DEER STAR Methods 的定义打分
#   C  compare_metrics.py：与 reference/metrics_full.json 比对
#
# ⛔ 两个会出错的点：
#   1. --max-length 必须 1024。发布数字是截断到 1024 的口径，而 embed.py 默认全长；
#      用别的长度会得到对不上的数，且不报错。
#   2. RELEASE_ROOT 不能靠 ${BASH_SOURCE[0]} 推 —— sbatch 把脚本拷到
#      /var/spool/slurm/slurmd/job<id> 再执行，那里只有脚本自己。
set -u

RELEASE_ROOT="${RELEASE_ROOT:?必须给 RELEASE_ROOT=<release 根目录绝对路径>}"
HERE="$RELEASE_ROOT/reproduce/ec_deer_figure7"
EC_HEAD="${EC_HEAD:-$RELEASE_ROOT/models/ec_head/ec_head.pt}"
WORK="${WORK:-${SLURM_SUBMIT_DIR:-$PWD}/fig7_${SLURM_JOB_ID:-local}}"

for f in "$RELEASE_ROOT/code/scripts/embed.py" "$HERE/score_figure7.py" \
         "$HERE/data/templates.fasta" "$HERE/data/database.fasta" \
         "$HERE/data/metadata.csv" "$EC_HEAD"; do
  [[ -e "$f" ]] || { echo "!! 找不到 $f" >&2; exit 1; }
done
mkdir -p "$WORK/encoded" "$WORK/out"

module load miniconda3
# ProtT5 的 tokenizer 需要 sentencepiece，请用装过它的环境
source activate "${SPLM_T5_ENV:?Set SPLM_T5_ENV to your conda environment}"
export TORCH_HOME="${TORCH_HOME:-$PWD/.torch_cache}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
unset SCOPE_EXTRA_PATH

echo "release : $RELEASE_ROOT"
echo "ec head : $EC_HEAD"
echo "work    : $WORK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ---------------------------------------------------------------- A 编码
for split in templates database; do
  if [[ -s "$WORK/encoded/$split.npz" ]]; then
    echo ">>> $split.npz 已存在，跳过编码"; continue
  fi
  echo ">>> 编码 $split（--modes tm，max-length 1024）"
  python "$RELEASE_ROOT/code/scripts/embed.py" \
      --fasta "$HERE/data/$split.fasta" \
      --modes tm --max-length 1024 --batch-size 32 \
      --out "$WORK/encoded/$split.npz" || exit 1
done

# ---------------------------------------------------------------- B 打分
echo ">>> 打分（score_figure7.py，自包含）"
python "$HERE/score_figure7.py" \
    --templates "$WORK/encoded/templates.npz" \
    --database  "$WORK/encoded/database.npz" \
    --ec-head   "$EC_HEAD" \
    --out       "$WORK/out" || exit 1

# ---------------------------------------------------------------- C 比对
echo ">>> 与发布数字比对"
python "$HERE/compare_metrics.py" "$WORK/out/metrics.json" \
                                  "$HERE/reference/metrics_full.json"
rc=$?

echo "exit code: $rc"
exit $rc
