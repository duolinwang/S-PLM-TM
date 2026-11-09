#!/usr/bin/env bash
# 一键准备 Pfam 预筛要用的东西（只有 --prefilter pfamfam/pfamclan 才需要）。
#
#   bash code/scripts/setup_pfam.sh [目标目录]
#
# 默认目标目录 = <release>/pfam_data，也就是 pfam.py / search.py 的默认查找位置，
# 装完之后 `--prefilter pfamclan` 不用再带任何路径参数。
#
# 它做四件事：
#   1. 检查 hmmpress / hmmscan / perl 在不在（缺了立刻说装什么，而不是跑一半失败）
#   2. 下载 **Pfam 36.0** 的 Pfam-A.hmm.gz (300 MB) 与 Pfam-A.hmm.dat.gz
#   3. hmmpress 建索引
#   4. 下载并解开 PfamScan
#
# ⚠️ 版本固定 36.0。已发表数字就是这一版跑的；换版本会让约 2% 的 clan 归属变化、
#    候选集随之改变，而且**不会报错**。理由与验证见 pfam_data/README.md。
# ⚠️ PfamScan 是 GPL-2.0+，所以不随包分发、在这里下载 —— 别把它 commit 进本包。
set -euo pipefail

PFAM_RELEASE="36.0"
BASE="https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam${PFAM_RELEASE}"
TOOLS="https://ftp.ebi.ac.uk/pub/databases/Pfam/Tools/PfamScan.tar.gz"

# 显式定位包根：sbatch 会把脚本拷到 /var/spool/slurm/... 去执行，
# 那时 $BASH_SOURCE 推不出包位置，所以允许用 RELEASE_ROOT 覆盖。
if [[ -n "${RELEASE_ROOT:-}" ]]; then
  ROOT="${RELEASE_ROOT}"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
DEST="${1:-${ROOT}/pfam_data}"

echo "release root : ${ROOT}"
echo "install into : ${DEST}"
echo "Pfam release : ${PFAM_RELEASE} (pinned — do not bump without re-reading pfam_data/README.md)"
echo

# ── 1. 前置检查 ─────────────────────────────────────────────────────────────
missing=()
for exe in hmmpress hmmscan perl; do
  command -v "$exe" >/dev/null 2>&1 || missing+=("$exe")
done
if (( ${#missing[@]} )); then
  echo "!! 缺少：${missing[*]}"
  echo "   装法（任选）："
  echo "     conda install -c bioconda hmmer perl-bioperl perl-moose"
  echo "     apt-get install hmmer bioperl"
  echo "   装好后重跑本脚本。"
  exit 1
fi
echo "ok: hmmpress / hmmscan / perl 都在"

DB="${DEST}/Pfam_db"
mkdir -p "${DB}" "${DEST}"

fetch() {   # fetch <url> <outfile>
  local url="$1" out="$2"
  if [[ -s "${out}" ]]; then
    echo "  have $(basename "${out}")，跳过下载"
    return
  fi
  echo "  downloading $(basename "${out}") ..."
  # -C - 支持断点续传：300 MB 断在半路不必从头再来
  curl -fL --retry 3 --retry-delay 5 -C - -o "${out}.part" "${url}"
  mv -f "${out}.part" "${out}"
}

# ── 2. HMM 库 ───────────────────────────────────────────────────────────────
echo
echo "[1/3] Pfam-A.hmm (约 300 MB 压缩)"
fetch "${BASE}/Pfam-A.hmm.gz"     "${DB}/Pfam-A.hmm.gz"
fetch "${BASE}/Pfam-A.hmm.dat.gz" "${DB}/Pfam-A.hmm.dat.gz"
for f in Pfam-A.hmm Pfam-A.hmm.dat; do
  if [[ ! -s "${DB}/${f}" ]]; then
    echo "  gunzip ${f}.gz ..."
    gunzip -c "${DB}/${f}.gz" > "${DB}/${f}"
  fi
done

# ── 3. hmmpress ─────────────────────────────────────────────────────────────
echo
echo "[2/3] hmmpress"
if [[ -s "${DB}/Pfam-A.hmm.h3f" ]]; then
  echo "  已有 .h3f，跳过"
else
  ( cd "${DB}" && hmmpress Pfam-A.hmm )
fi

# ── 4. PfamScan ─────────────────────────────────────────────────────────────
echo
echo "[3/3] PfamScan"
if [[ -s "${DEST}/PfamScan/pfam_scan.pl" ]]; then
  echo "  已有 pfam_scan.pl，跳过"
else
  fetch "${TOOLS}" "${DEST}/PfamScan.tar.gz"
  tar -xzf "${DEST}/PfamScan.tar.gz" -C "${DEST}"
  rm -f "${DEST}/PfamScan.tar.gz"
fi

# ── 验收 ────────────────────────────────────────────────────────────────────
echo
ok=1
for f in "${DEST}/Pfam-A.clans.tsv" "${DB}/Pfam-A.hmm" "${DB}/Pfam-A.hmm.h3f" \
         "${DEST}/PfamScan/pfam_scan.pl"; do
  if [[ -s "${f}" ]]; then
    echo "  OK   ${f}"
  else
    echo "  MISSING  ${f}"
    ok=0
  fi
done

if (( ok )); then
  cat <<EOF

Pfam 预筛就绪。现在这样跑（不用再带路径参数）：

  python code/scripts/search.py --query q.npz --target t.npz \\
      --query-fasta queries.fasta --target-fasta targets.fasta \\
      --prefilter pfamclan --out hits.tsv
EOF
else
  echo
  echo "!! 没装全，上面标 MISSING 的那几项要查一下。"
  exit 1
fi
