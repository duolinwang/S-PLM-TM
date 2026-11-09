#!/bin/bash
# 把 models/ 下的权重传到 Hugging Face（private）。
#
#   export HF_REPO=<你的HF用户名>/S-PLM-TM
#   bash hf/upload_to_hf.sh
#
# 先决条件：
#   pip install "huggingface_hub[cli]"
#   hf auth login                 # 需要一个有 write 权限的 token
#
# ⚠️ 默认建 **private** 仓库。投稿时转 public 用：
#   hf repo settings $HF_REPO --private false
#   （或在网页 Settings 里切，可见性随时可改、改名/转移会自动重定向）
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
REPO="${HF_REPO:?请先 export HF_REPO=<用户名>/S-PLM-TM}"
MODELS="${SPLM_TM_MODELS:-$ROOT/models}"

# hf 的可执行文件常常不在 PATH 上（Windows 装在 %APPDATA%\Python\PythonXY\Scripts，
# Git Bash 看不到；Linux 上则可能在 ~/.local/bin）。能直接调就直接调，否则回退到
# `python -m`，功能完全一样，省得在这一步卡住。
if command -v hf >/dev/null 2>&1; then
  HF=(hf)
elif python -m huggingface_hub.cli.hf --version >/dev/null 2>&1; then
  HF=(python -m huggingface_hub.cli.hf)
  echo ">>> hf 不在 PATH 上，改用 python -m huggingface_hub.cli.hf"
else
  echo "!! 没装 huggingface_hub：pip install 'huggingface_hub[cli]'" >&2
  exit 1
fi

"${HF[@]}" auth whoami >/dev/null 2>&1 || {
  echo "!! 还没登录。先跑：${HF[*]} auth login   （需要一个有 write 权限的 token）" >&2
  exit 1
}

# ⭐ 只传**二进制权重**。配套的 yaml 与训练日志是文本，归 git 管 ——
# 它们在 GitHub 仓库的 models/ 下，有版本历史也能看 diff。
# ⛔ 别把 yaml 也传过来：它的职责是与 checkpoint 严格配对，配错不会报错、只会静默
#    产出错向量。两边各存一份就会漂移，而且用户先 git clone 再
#    `hf download --local-dir models/` 时，HF 那份会直接盖掉 git 那份，谁后到谁赢。
#    我们自己的代码也只读 git 那份（paths.py 在 models/backbone/ 下找）。
FILES=(
  "backbone/splm_tm_backbone.pth"
  "tm_head/tm_calib_head.pt"
  "ec_head/ec_head.pt"
  "align_head/align_head.pt"
  "align_head/alternatives/align_head_published_table.pt"
)

missing=0
for f in "${FILES[@]}"; do
  if [[ ! -f "$MODELS/$f" ]]; then echo "!! 缺文件: $MODELS/$f" >&2; missing=1; fi
done
[[ $missing -eq 0 ]] || { echo "   先把权重准备齐（见 models/DOWNLOAD.md）再传。" >&2; exit 1; }

echo ">>> 目标 repo : $REPO (private)"
echo ">>> 本地 models: $MODELS"
du -ch "${FILES[@]/#/$MODELS/}" 2>/dev/null | tail -1

"${HF[@]}" repo create "$REPO" --repo-type model --private -y 2>/dev/null \
  || echo ">>> repo 已存在，继续上传"

# model card 先传，这样仓库页面不会是空的
"${HF[@]}" upload "$REPO" "$HERE/README.md" README.md --repo-type model
# ⭐ 许可必须跟着权重走：仓库是 public，而 LICENSE 全文在 GitHub 那边（私有库，
#    外人点进去 404）。非商用条款要生效，拿到权重的人就得能看到条款。
"${HF[@]}" upload "$REPO" "$ROOT/LICENSE" LICENSE --repo-type model

for f in "${FILES[@]}"; do
  echo ">>> 上传 $f"
  "${HF[@]}" upload "$REPO" "$MODELS/$f" "$f" --repo-type model
done

echo
echo ">>> 完成。下载侧自检："
echo "    ${HF[*]} download $REPO --local-dir /tmp/splm_check"
echo "    md5sum /tmp/splm_check/ec_head/ec_head.pt   # 应为 e306df5929c804702a7fb5078ca267f6"
