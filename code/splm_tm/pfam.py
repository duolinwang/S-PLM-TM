# -*- coding: utf-8 -*-
"""Pfam 预筛 —— 复现 MODEL_CARD 里「+TM head +Pfam 预筛」那一档部署口径。

为什么 release 需要这个
----------------------
`models/MODEL_CARD.md` 报的 SCOPe 远同源与细菌 EC 两组数字，是在 **Pfam 预筛**下测的
（和 PLMSearch 的部署口径对齐）。但发布包原先只有 `embed.py`（序列 -> 向量），
既没有检索驱动也没有预筛，于是**拿 release 复现不出那些数字**。这个模块补上缺口。

预筛做什么
----------
Pfam 注释与 embedding **完全独立**：PfamScan 只吃序列，编码器也只吃序列，谁都不读
对方的输出。两者只在打分时汇合 —— 先算完整的 query x target 相似度矩阵，再按行把
「与 query 不共享任何 Pfam family/clan」的 target 置成 -inf。

    queries.fasta --PfamScan--> query Pfam --+
                                             +--> overlap map --+
    targets.fasta --PfamScan--> target Pfam -+                  v
                 \\--S-PLM-TM--> embedding --> 相似度矩阵 --> mask(-inf) --> 排名

⚠️ 所以它**不省编码时间**，只改排名。PLMSearch 原版把它当真预筛用来省时间；这里保持
   与我们发表数字一致的语义（算完再 mask），不要改成"先筛后算"，那会得到不同的数。

⚠️ 一个容易被忽略的行为：某个 query **一个 Pfam 都没命中**时，它那一行**不加 mask**，
   退回全库检索。所以 pfamclan 口径下的结果其实是「有注释的走预筛 + 没注释的走全库」
   的混合，不是纯预筛。这与产出已发表数字的实现一致，不要"修"。

本模块与研究仓库 `microbiom_searchproblem/bacteria_benchmark/bacteria_pfam.py` 的
逻辑逐位等价，用 `code/scripts/check_pfam_parity.py` 核验。

要用到的数据
------------
    pfam_data/Pfam-A.clans.tsv          **已随包**（Pfam 36.0），family -> clan
    pfam_data/Pfam_db/Pfam-A.hmm        300 MB，`bash code/scripts/setup_pfam.sh` 下
    pfam_data/PfamScan/pfam_scan.pl     同上（GPL-2.0+，故不随包）
后两者跑一次 setup 脚本就行，之后 `--prefilter pfamclan` 不用带任何路径参数。

⚠️ **clans.tsv 钉在 Pfam 36.0**，因为 Pfam 在版本之间会重组 clan：拿已发表那次
   PfamScan 输出的 clan 列比对，36.0 零分歧、37.0 差 1 对、当前版差 15 对（878 个
   不重复断言里）。换版本会让候选集变化、复现数字对不上**且不报错**。
   验证过程与数据见 `pfam_data/README.md`。
"""
from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set

import numpy as np

# 随包/一键装好之后的默认位置（setup_pfam.sh 就装成这个结构）
#   <release>/pfam_data/Pfam-A.clans.tsv      随包，Pfam 36.0
#   <release>/pfam_data/Pfam_db/              setup_pfam.sh 下载 + hmmpress
#   <release>/pfam_data/PfamScan/pfam_scan.pl setup_pfam.sh 下载
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent      # <release>/
DEFAULT_PFAM_DATA_ROOT = _PKG_ROOT / "pfam_data"
DEFAULT_CLAN_TSV = DEFAULT_PFAM_DATA_ROOT / "Pfam-A.clans.tsv"

__all__ = [
    "DEFAULT_PFAM_DATA_ROOT",
    "DEFAULT_CLAN_TSV",
    "parse_pfam_scan_txt",
    "run_pfam_scan",
    "load_clan_map",
    "load_pfam_sets",
    "build_pfam_overlap_map",
    "apply_pfam_mask",
    "apply_pfam_mask_row",
    "prepare_pfam_overlap",
    "pfam_missing_message",
]


def pfam_missing_message(pfam_data_root) -> str:
    return (
        "Pfam 预筛要用的 HMM 库与 PfamScan 没找到。跑一次就好：\n"
        "    bash code/scripts/setup_pfam.sh\n"
        "它会下载 Pfam 36.0 的 HMM 库（约 300 MB）与 PfamScan 并 hmmpress，\n"
        "装成下面这个结构（也就是默认查找位置）：\n"
        "    %s/Pfam_db/Pfam-A.hmm      (+ hmmpress 的 .h3*)\n"
        "    %s/PfamScan/pfam_scan.pl\n"
        "前置：conda install -c bioconda hmmer perl-bioperl perl-moose\n"
        "（family -> clan 的 Pfam-A.clans.tsv 已随包，不用下。）\n"
        "另两条出路：已有别处算好的 Pfam JSON 就用 --query-pfam-json/--target-pfam-json\n"
        "绕过 PfamScan；或者干脆不预筛（去掉 --prefilter），那是另一档口径。"
        % (pfam_data_root, pfam_data_root)
    )


# ───────────────────────────── 注释解析 ─────────────────────────────────────
def parse_pfam_scan_txt(txt_path: Path, seq_ids: Sequence[str]) -> Dict[str, List[str]]:
    """pfam_scan.pl 的文本输出 -> {seq_id: [pfam_acc, ...]}。

    输出是空格分隔的：
        <seq_id> <aln_start> <aln_end> <env_start> <env_end> <hmm_acc> <hmm_name> ...
    取第 0 列 seq_id 与第 5 列 Pfam accession（形如 PF00069.27）。
    没有命中的序列保留空列表（而不是缺键），下游靠这个区分"没注释"和"不在集合里"。
    """
    result: Dict[str, List[str]] = {sid: [] for sid in seq_ids}
    if not Path(txt_path).exists():
        return result
    with Path(txt_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            seq_id, pfam_acc = parts[0], parts[5]
            if seq_id in result:
                result[seq_id].append(pfam_acc)
    return result


def run_pfam_scan(fasta_path: Path, json_path: Path, pfam_data_root: Path) -> None:
    """对一个 FASTA **整体**跑一次 pfam_scan.pl，结果存成 JSON。

    注意是整文件批量跑一次，不是每条序列跑一次 —— 那会慢到不可用。
    """
    fasta_path, json_path = Path(fasta_path), Path(json_path)
    pfam_data_root = Path(pfam_data_root)
    scan_pl = pfam_data_root / "PfamScan" / "pfam_scan.pl"
    db_dir = pfam_data_root / "Pfam_db"
    if not scan_pl.is_file() or not db_dir.is_dir():
        raise FileNotFoundError(pfam_missing_message(pfam_data_root))

    json_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_txt = json_path.parent / ("_pfam_tmp_%s.txt" % json_path.stem)

    print("  [PfamScan] %s ..." % fasta_path.name)
    subprocess.run(
        ["perl", str(scan_pl), "-fasta", str(fasta_path),
         "-dir", str(db_dir), "-outfile", str(tmp_txt)],
        check=True,
    )

    seq_ids: List[str] = []
    with fasta_path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                seq_ids.append(line[1:].strip().split()[0])

    result = parse_pfam_scan_txt(tmp_txt, seq_ids)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f)
    print("  [PfamScan] saved %s (%d sequences)" % (json_path, len(result)))


def load_clan_map(clan_tsv: Path) -> Dict[str, str]:
    """Pfam-A.clans.tsv -> {family_acc: clan_acc}。

    列：family_acc  clan_acc  clan_id  family_id  family_description
    clan_acc 为空表示该 family 不属于任何 clan，这类不入表。
    """
    fam_to_clan: Dict[str, str] = {}
    with Path(clan_tsv).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                fam, clan = parts[0].strip(), parts[1].strip()
                if clan:
                    fam_to_clan[fam] = clan
    return fam_to_clan


def load_pfam_sets(
    json_path: Path,
    use_clan: bool,
    fam_to_clan: Optional[Mapping[str, str]] = None,
) -> Dict[str, Set[str]]:
    """Pfam JSON -> {seq_id: set(accession)}。

    use_clan=False -> family accession 集合（PF00069）
    use_clan=True  -> clan accession 集合（CL0016）；**没有 clan 的 family 保留
                      family accession**，这样它仍能与同 family 的 target 匹配上，
                      而不是变成"没注释"去走全库回退。
    版本号一律剥掉：PF00069.27 -> PF00069。
    """
    with Path(json_path).open(encoding="utf-8") as f:
        raw: Dict[str, List[str]] = json.load(f)

    result: Dict[str, Set[str]] = {}
    for seq_id, accs in raw.items():
        stripped = {a.split(".")[0] for a in accs}
        if use_clan and fam_to_clan:
            result[seq_id] = {fam_to_clan.get(fam, fam) for fam in stripped}
        else:
            result[seq_id] = stripped
    return result


# ───────────────────────────── 重叠图与掩码 ─────────────────────────────────
def build_pfam_overlap_map(
    query_pfam: Mapping[str, Set[str]],
    target_pfam: Mapping[str, Set[str]],
    target_ids: Sequence[str],
) -> Dict[str, Set[str]]:
    """{query_id: 与它共享 >=1 个 family/clan 的 target_id 集合}。"""
    target_id_set = set(target_ids)
    inv: Dict[str, Set[str]] = defaultdict(set)
    for tid, fams in target_pfam.items():
        if tid not in target_id_set:
            continue
        for fam in fams:
            inv[fam].add(tid)

    overlap: Dict[str, Set[str]] = {}
    for qid, qfams in query_pfam.items():
        cands: Set[str] = set()
        for fam in qfams:
            cands.update(inv.get(fam, ()))
        overlap[qid] = cands
    return overlap


def apply_pfam_mask(
    sim_matrix: np.ndarray,
    query_ids: Sequence[str],
    target_ids: Sequence[str],
    overlap_map: Mapping[str, Set[str]],
    *,
    verbose: bool = True,
) -> np.ndarray:
    """逐行掩码相似度矩阵，返回同形状的新矩阵（不原地改）。

    每一行：有候选 -> 非候选置 -inf；**没有候选 -> 整行原样保留（全库回退）**。
    """
    sim = np.asarray(sim_matrix)
    masked = np.copy(sim)
    target_id_to_idx = {tid: i for i, tid in enumerate(target_ids)}
    n_targets = len(target_ids)

    n_masked = n_fallback = 0
    for i, qid in enumerate(query_ids):
        cand_ids = overlap_map.get(str(qid), set())
        cand_idxs = {target_id_to_idx[t] for t in cand_ids if t in target_id_to_idx}
        if not cand_idxs:
            n_fallback += 1
            continue
        row = np.full(n_targets, -np.inf, dtype=sim.dtype)
        for j in cand_idxs:
            row[j] = sim[i, j]
        masked[i] = row
        n_masked += 1

    if verbose:
        print("  [Pfam mask] masked %d queries | full-library fallback %d queries"
              % (n_masked, n_fallback))
    return masked


def apply_pfam_mask_row(
    scores_row: np.ndarray,
    query_id: str,
    target_ids: Sequence[str],
    overlap_map: Mapping[str, Set[str]],
) -> np.ndarray:
    """单行版本，供逐 query 打分的流程用。"""
    row = np.asarray(scores_row, dtype=np.float64).reshape(-1)
    cand_ids = overlap_map.get(str(query_id), set())
    cand_idxs = {i for i, tid in enumerate(target_ids) if tid in cand_ids}
    if not cand_idxs:
        return row.copy()
    masked = np.full(row.shape, -np.inf, dtype=np.float64)
    for j in cand_idxs:
        masked[j] = row[j]
    return masked


# ───────────────────────────── 一条龙 ───────────────────────────────────────
def prepare_pfam_overlap(
    query_fasta: Path,
    target_fasta: Path,
    target_ids: Sequence[str],
    pfam_data_root: Optional[Path],
    cache_dir: Path,
    use_clan: bool,
    clan_tsv: Optional[Path] = None,          # None -> 用随包的 Pfam 36.0 那份
    *,
    query_json: Optional[Path] = None,
    target_json: Optional[Path] = None,
) -> Dict[str, Set[str]]:
    """跑 PfamScan（或直接用已有 JSON）并建重叠图。JSON 会缓存，不会重跑。

    已经有别处算好的 Pfam JSON 时，用 query_json / target_json 直接传进来，
    这时 pfam_data_root 可以是 None（不需要装 PfamScan）。
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    mode = "clan" if use_clan else "fam"

    q_json = Path(query_json) if query_json else cache_dir / ("queries_%s_pfam.json" % mode)
    t_json = Path(target_json) if target_json else cache_dir / ("targets_%s_pfam.json" % mode)

    for fasta, js, what in ((query_fasta, q_json, "query"),
                            (target_fasta, t_json, "target")):
        if js.is_file():
            print("  [Pfam] cached %s JSON: %s" % (what, js))
            continue
        if pfam_data_root is None:
            raise FileNotFoundError(
                "%s 不存在，且没有给 --pfam-data-root，无法现跑 PfamScan。\n%s"
                % (js, pfam_missing_message("<pfam_data_root>")))
        run_pfam_scan(Path(fasta), js, Path(pfam_data_root))

    fam_to_clan = None
    if use_clan:
        if clan_tsv is None:
            clan_tsv = DEFAULT_CLAN_TSV          # 随包的 Pfam 36.0
        if not Path(clan_tsv).is_file():
            raise FileNotFoundError(
                "clan 模式需要 Pfam-A.clans.tsv，但这个路径不存在：%s\n"
                "随包应当自带 %s；被删掉的话从 "
                "https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam36.0/"
                "Pfam-A.clans.tsv.gz 取（⚠️ 必须是 36.0，理由见 pfam_data/README.md）"
                % (clan_tsv, DEFAULT_CLAN_TSV))
        fam_to_clan = load_clan_map(Path(clan_tsv))
        print("  [Pfam] clan map: %d families -> clans" % len(fam_to_clan))

    q_pfam = load_pfam_sets(q_json, use_clan, fam_to_clan)
    t_pfam = load_pfam_sets(t_json, use_clan, fam_to_clan)
    overlap = build_pfam_overlap_map(q_pfam, t_pfam, target_ids)

    n_with = sum(1 for v in overlap.values() if v)
    avg = float(np.mean([len(v) for v in overlap.values()])) if overlap else 0.0
    print("  [Pfam %s] %d queries | %d with candidates | avg candidates %.1f / %d targets"
          % (mode, len(overlap), n_with, avg, len(target_ids)))
    return overlap
