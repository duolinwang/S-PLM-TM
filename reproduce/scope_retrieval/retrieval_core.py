"""SCOPE 检索指标（sensitivity / AUC-1FP），供 scope_metrics 与 scope_mmseqs2 共用。"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

# 与 scope_metrics 评测一致；修改时请保持两处行为对齐。
TOP_K = 200


def mmseqs2_sensitivity_filename_suffix(top_k: int) -> str:
    """MMseqs2 输出 / 合并用 TSV 文件名中段：top{K}（与 sensitivity@TOP_K 一致）。"""
    return f"top{top_k}"


def _auc_1fp_discrete_roc_normalized(
    scores_row: np.ndarray,
    q_idx: int,
    pos: np.ndarray,
    ignored: np.ndarray | None = None,
) -> float:
    """AUC-1FP：按分降序扫到第一个 fp_eligible 时，已遇到的 pos 个数 / P（P=pos.sum()，含 self）。

    ignored[j]：软负例，不进入 fp_eligible（见 scope_metrics 模块文档）。
    """
    n = len(scores_row)
    p = int(pos.sum())
    if p <= 0:
        return float("nan")
    ign = (
        np.zeros(n, dtype=bool) if ignored is None else np.asarray(ignored, dtype=bool)
    )
    fp_eligible = (np.arange(n, dtype=np.int32) != q_idx) & ~pos & ~ign
    n_neg = int(fp_eligible.sum())
    if n_neg <= 0:
        return 1.0
    order = np.lexsort((np.arange(n), -scores_row))
    tp_before = 0
    for j in order:
        if fp_eligible[j]:
            return float(tp_before) / float(p)
        if pos[j]:
            tp_before += 1
    return 1.0


def per_query_sensitivity(
    scores_row: np.ndarray,
    query_key: str,
    query_label: str,
    db_domains: Sequence[str],
    db_labels: np.ndarray,
    dom_index: dict[str, int],
    top_k: int,
) -> float:
    if query_key not in dom_index or not query_label:
        return float("nan")
    pos = (db_labels == query_label) & (db_labels != "")
    n_pos = int(pos.sum())
    if n_pos <= 0:
        return float("nan")
    k_take = min(top_k, len(scores_row))
    order = np.argsort(-scores_row)[:k_take]
    return int(pos[order].sum()) / n_pos


def _auc_ignored_same_soft_class(
    q_idx: int, pos: np.ndarray, soft_labels: np.ndarray
) -> np.ndarray:
    """构造 ignored：同 superfamily（family 任务）或同 fold（sf 任务）、且非严格正例、非 query 自身。"""
    n = len(pos)
    qs = str(soft_labels[q_idx]).strip()
    if not qs or qs.lower() == "nan":
        return np.zeros(n, dtype=bool)
    sl = soft_labels.astype(str)
    return (np.arange(n) != q_idx) & ~pos & (sl == qs)


def per_query_auc_1fp(
    scores_row: np.ndarray,
    query_key: str,
    query_label: str,
    db_domains: Sequence[str],
    db_labels: np.ndarray,
    dom_index: dict[str, int],
    auc_soft_class_labels: np.ndarray | None = None,
) -> float:
    """单 query 的 auc_1fp：调用 _auc_1fp_discrete_roc_normalized。"""
    if query_key not in dom_index or not str(query_label).strip():
        return float("nan")
    q_idx = dom_index[query_key]
    n = len(scores_row)
    pos = (db_labels == query_label) & (db_labels != "")
    ignored = (
        None
        if auc_soft_class_labels is None
        else _auc_ignored_same_soft_class(q_idx, pos, auc_soft_class_labels)
    )
    return _auc_1fp_discrete_roc_normalized(scores_row, q_idx, pos, ignored)


def per_query_n_same_label_in_db(
    query_key: str,
    query_label: str,
    db_labels: np.ndarray,
    dom_index: dict[str, int],
) -> float:
    """库中与 query 同标签且标签非空的 domain 总数（含 query 自身）。"""
    if query_key not in dom_index or not str(query_label).strip():
        return float("nan")
    lab = str(query_label).strip()
    mask = (db_labels == lab) & (db_labels != "")
    return float(int(mask.sum()))


def _collect_metrics_impl(
    sim_rows: List[np.ndarray],
    queries: Sequence[str],
    q_labels: np.ndarray,
    db_dom: Sequence[str],
    db_labels: np.ndarray,
    dom_index: dict[str, int],
    top_k: int,
    auc_soft_class_labels: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(sim_rows)
    sens, aucs, n_lib = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        sens[i] = per_query_sensitivity(
            sim_rows[i],
            queries[i],
            q_labels[i],
            db_dom,
            db_labels,
            dom_index,
            top_k,
        )
        aucs[i] = per_query_auc_1fp(
            sim_rows[i],
            queries[i],
            q_labels[i],
            db_dom,
            db_labels,
            dom_index,
            auc_soft_class_labels,
        )
        n_lib[i] = per_query_n_same_label_in_db(
            queries[i], q_labels[i], db_labels, dom_index
        )
    return sens, aucs, n_lib


def collect_family_retrieve_metrics(
    sim_rows: List[np.ndarray],
    queries: Sequence[str],
    q_family_sunid: np.ndarray,
    db_dom: Sequence[str],
    db_family_sunid: np.ndarray,
    db_superfamily_sccs: np.ndarray,
    dom_index: dict[str, int],
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Family 任务：pos = 同 family sccs；软负例 = 同 superfamily、不同 family（不计 FP）。"""
    return _collect_metrics_impl(
        sim_rows,
        queries,
        q_family_sunid,
        db_dom,
        db_family_sunid,
        dom_index,
        top_k,
        db_superfamily_sccs,
    )


def collect_superfamily_retrieve_metrics(
    sim_rows: List[np.ndarray],
    queries: Sequence[str],
    q_superfamily_sccs: np.ndarray,
    db_dom: Sequence[str],
    db_superfamily_sccs: np.ndarray,
    db_fold_sccs: np.ndarray,
    dom_index: dict[str, int],
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Superfamily 任务：pos = 同 superfamily sccs；软负例 = 同 fold、不同 superfamily（不计 FP）。"""
    return _collect_metrics_impl(
        sim_rows,
        queries,
        q_superfamily_sccs,
        db_dom,
        db_superfamily_sccs,
        dom_index,
        top_k,
        db_fold_sccs,
    )
