#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""核验 release 复刻的 EC head 与原实现**逐位一致**。

本 release 的 `splm_tm.ECRetrievalHead` 是把原实现
`remote-search/src/experiments/exp_3-1.1/heads_lib.py` 里的 `_MLPBlock` / `MLPHead`
抄了一份（为了不依赖那个目录）。抄错一个细节（残差加不加、LayerNorm 在前还是在后、
dropout 位置）**不会报错**，只会静默产出不同的向量。所以这个脚本存在：
拿同一个 `head.pt`、同一批输入，比较两边的输出。

用法（在能读到原实现目录的机器上，例如 Hellbender）::

    python code/scripts/check_ec_head_parity.py
    python code/scripts/check_ec_head_parity.py --head <别的 head.pt>

判据：max |diff| == 0（同一套权重、同一组算子，应当是逐位相同，不是"近似相等"）。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from splm_tm import ECRetrievalHead                      # noqa: E402
from splm_tm import paths as splm_paths                  # noqa: E402

DEFAULT_HEADS_LIB = os.environ.get("SPLM_TM_HEADS_LIB", "")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--head", default=None,
                   help="要核验的 head .pt（默认用 models/ec_head/ 下解析到的那个）")
    p.add_argument("--heads-lib", default=DEFAULT_HEADS_LIB,
                   help="原实现的 exp_3-1.1 目录（heads_lib.py 在里面）")
    p.add_argument("-n", type=int, default=64, help="随机输入条数")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    head_path = Path(args.head) if args.head else splm_paths.default_ec_head()
    print(f">>> head     : {head_path}")
    print(f">>> heads_lib: {args.heads_lib}")

    hl_dir = Path(args.heads_lib)
    if not (hl_dir / "heads_lib.py").is_file():
        raise SystemExit(
            f"{hl_dir} 下没有 heads_lib.py —— 这个核验只能在能读到原实现目录的机器上跑"
            f"（用 --heads-lib 指对路径）。")
    sys.path.insert(0, str(hl_dir))
    # heads_lib 顶层只有一个外部 import：`from residue_head_common import
    # masked_mean_pool, reverse_padded`，那是给 transformer/lstm/mamba 这些**残基级**头用的，
    # MLPHead 一行都不碰。而 residue_head_common 会连锁 import 整条评测/数据管线
    # （evaluate_head -> data -> dataset_train -> metrics ...），在这里既装不起来也没必要。
    # 所以先塞一个 stub 进 sys.modules，让 heads_lib 顶层过去。
    if "residue_head_common" not in sys.modules:
        try:
            import residue_head_common                     # noqa: F401
        except Exception as e:                             # noqa: BLE001
            import types
            stub = types.ModuleType("residue_head_common")

            def _unavailable(*_a, **_k):
                raise RuntimeError("stub：本核验只覆盖 mlp 族，不涉及残基级头的池化")

            stub.masked_mean_pool = _unavailable
            stub.reverse_padded = _unavailable
            sys.modules["residue_head_common"] = stub
            print(f">>> residue_head_common 装不起来（{type(e).__name__}: "
                  f"{str(e).splitlines()[0]}），已用 stub 顶替 —— MLPHead 不用它")
    import heads_lib                                      # noqa: E402

    ck = torch.load(str(head_path), map_location="cpu")
    cfg, sd = dict(ck.get("cfg") or {}), ck["state_dict"]
    in_dim = int(sd["fc1.weight"].shape[1])

    ref = heads_lib.build_head(ck.get("family", "mlp"), cfg, in_dim)
    ref.load_state_dict(sd)
    ref.eval()

    mine = ECRetrievalHead.load(head_path, map_location="cpu")

    # 结构对照
    print(f">>> 参数量 ref={sum(p.numel() for p in ref.parameters()):,}  "
          f"mine={sum(p.numel() for p in mine.parameters()):,}")
    ref_keys, my_keys = sorted(ref.state_dict()), sorted(mine.state_dict())
    if ref_keys != my_keys:
        print("!! state_dict 键不一致")
        print(f"   只在 ref : {sorted(set(ref_keys) - set(my_keys))}")
        print(f"   只在 mine: {sorted(set(my_keys) - set(ref_keys))}")
        return 1
    print(f">>> state_dict 键一致（{len(ref_keys)} 个）")
    print(f">>> 合并后的 cfg: {mine.cfg}")

    # 数值对照
    g = torch.Generator().manual_seed(args.seed)
    x = torch.randn(args.n, in_dim, generator=g)
    x = torch.nn.functional.normalize(x, dim=-1)       # 模拟 emb_h（已 L2 归一化）
    with torch.no_grad():
        a, b = ref(x), mine(x)
    diff = (a - b).abs().max().item()
    print(f">>> 输出形状 ref={tuple(a.shape)} mine={tuple(b.shape)}")
    print(f">>> max |diff| = {diff:.3e}")

    if diff == 0.0:
        print("\n✅ 逐位一致 —— release 的复刻与原实现等价")
        return 0
    if diff < 1e-6:
        print(f"\n⚠️ 不是逐位相同但在 {diff:.1e} 以内。同一套算子本应完全相同，"
              f"出现差异说明算子顺序或类型有细微不同，建议排查。")
        return 0
    print("\n❌ 输出不一致 —— 复刻有错，不要用 release 的 EC head 出数")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
