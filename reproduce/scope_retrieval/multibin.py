"""Shared sequence-similarity-binned SCOPe retrieval evaluation."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from retrieval_core import TOP_K

DEFAULT_SEQ_ID_BINS = "0,10,20,30,40,50,60,70,80,90,100"
DEFAULT_MULTIBIN_TOP_K_RUNS = (1, 5, 10, 20, 50, 100, TOP_K)
DEFAULT_SIMILARITY_CACHE_DIR = Path(__file__).resolve().parent / "scope_similarity_cache"
DEFAULT_SEQ_ID_CACHE = (
    DEFAULT_SIMILARITY_CACHE_DIR / "scope_query_positive_seq_identity_nw_blosum62.tsv"
)
SEQ_IDENTITY_METHOD = "biopython_pairwisealigner_global_blosum62_gapopen10_gapextend0.5"
MULTIBIN_EXCEL_SHEETS = ("sequence_family", "sequence_superfamily")
MULTIBIN_COUNTS_SHEET = "multibin_counts"
MULTIBIN_COUNTS_HEADER = (
    "model",
    "task",
    "top_k",
    "bin",
    "n_queries",
    "n_positive_pairs",
)
MULTIBIN_COUNT_TASK_ORDER = ("family", "superfamily")

# Per-query AUC-1FP distribution within each sequence-identity bin (long format,
# one row per query) plus a companion summary-stats sheet (mean/std/var/quartiles).
SEQ_AUC_DIST_SHEET = "seq_auc_distribution"
SEQ_AUC_DIST_HEADER = ("model", "task", "bin", "query_id", "AUC-1FP")
SEQ_AUC_STATS_SHEET = "seq_auc_stats"
SEQ_AUC_STATS_HEADER = (
    "model",
    "task",
    "bin",
    "n_queries",
    "mean",
    "std",
    "var",
    "min",
    "p25",
    "median",
    "p75",
    "max",
)


def dated_multibin_excel_name(method_stem: str) -> str:
    date_tag = datetime.now().strftime("%Y%m%d")
    return f"{method_stem}_seq_multibin_auc1fp_{date_tag}.xlsx"


def multibin_metrics_excel_stem(prefilter: str, *, full_stem: str, prefilter_stem: str) -> str:
    """Excel 文件名 stem：full 用无 Pfam 后缀的 base（如 scope_tmvec），否则用 prefilter_stem。"""
    return full_stem if prefilter == "full" else prefilter_stem


def parse_edges(spec: str) -> list[float]:
    vals = [float(x.strip()) for x in str(spec).split(",") if x.strip()]
    if len(vals) < 2:
        raise ValueError("bin edges need at least two values")
    if any(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)):
        raise ValueError("bin edges must be strictly increasing")
    return vals


def bin_label(edges: Sequence[float], i: int) -> str:
    a, b = edges[i], edges[i + 1]
    return f"{a:g}-{b:g}"


def value_to_bin(v: float, edges: Sequence[float]) -> str | None:
    if not np.isfinite(v):
        return None
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == len(edges) - 2:
            if lo <= v <= hi:
                return bin_label(edges, i)
        elif lo <= v < hi:
            return bin_label(edges, i)
    return None


def find_col(fieldnames: Sequence[str], choices: Sequence[str]) -> str:
    lower = {c.lower(): c for c in fieldnames}
    for c in choices:
        if c.lower() in lower:
            return lower[c.lower()]
    raise ValueError(f"TSV missing one of columns: {choices}; saw {list(fieldnames)}")


def fmt(v: float) -> str:
    return f"{float(v):.10g}" if np.isfinite(v) else ""


def bin_display_label(label: str) -> str:
    text = str(label).strip()
    if "-" not in text:
        return text
    lo, hi = text.split("-", 1)
    close = "]" if hi.strip() == "100" else ")"
    return f"[{lo.strip()},{hi.strip()}{close}"


def expected_labeled_pairs(
    *,
    queries: Sequence[str],
    q_labels: np.ndarray,
    db_domains: Sequence[str],
    db_labels: np.ndarray,
) -> list[tuple[str, str]]:
    label_to_targets: dict[str, list[str]] = defaultdict(list)
    for d, lab in zip(db_domains, db_labels):
        labs = str(lab).strip()
        if labs:
            label_to_targets[labs].append(str(d).strip())

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for q, lab in zip(queries, q_labels):
        qs = str(q).strip()
        labs = str(lab).strip()
        if not qs or not labs:
            continue
        for t in label_to_targets.get(labs, []):
            key = (qs, str(t).strip())
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    return pairs


def _cached_pairs_present(
    path: Path,
    *,
    expected: Sequence[tuple[str, str]],
    symmetric: bool,
) -> tuple[bool, int]:
    if not path.is_file():
        return False, len(expected)
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            if reader.fieldnames is None:
                return False, len(expected)
            q_col = find_col(reader.fieldnames, ("query_id", "query", "q", "seq_id_1", "id1"))
            t_col = find_col(reader.fieldnames, ("target_id", "target", "t", "seq_id_2", "id2"))
            seen = {
                (str(r.get(q_col, "")).strip(), str(r.get(t_col, "")).strip())
                for r in reader
            }
    except Exception:
        return False, len(expected)

    missing = 0
    for q, t in expected:
        if (q, t) in seen:
            continue
        if symmetric and (t, q) in seen:
            continue
        missing += 1
    return missing == 0, missing


def make_global_aligner():
    try:
        from Bio.Align import PairwiseAligner
        from Bio.Align import substitution_matrices
    except ImportError as e:
        raise RuntimeError(
            "Biopython is required to compute missing sequence-similarity caches."
        ) from e

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    return aligner


def global_alignment_identity_percent(a: str, b: str, aligner) -> float:
    a = str(a).strip().upper()
    b = str(b).strip().upper()
    if not a or not b:
        return float("nan")
    if a == b:
        return 100.0

    aln = aligner.align(a, b)[0]
    coords = np.asarray(aln.coordinates, dtype=np.int64)
    matches = 0
    aligned_len = 0
    for k in range(coords.shape[1] - 1):
        a0, a1 = int(coords[0, k]), int(coords[0, k + 1])
        b0, b1 = int(coords[1, k]), int(coords[1, k + 1])
        da, db = a1 - a0, b1 - b0
        aligned_len += max(da, db)
        if da > 0 and db > 0:
            matches += sum(x == y for x, y in zip(a[a0:a1], b[b0:b1]))
    if aligned_len <= 0:
        return float("nan")
    return 100.0 * float(matches) / float(aligned_len)


def ensure_seq_identity_tsv_for_benchmark_queries(
    path: Path,
    *,
    family_queries: Sequence[str],
    q_family: np.ndarray,
    superfamily_queries: Sequence[str],
    q_superfamily: np.ndarray,
    db_domains: Sequence[str],
    db_family: np.ndarray,
    db_superfamily: np.ndarray,
    seq_d: Mapping[str, str],
) -> Path:
    path = path.resolve()
    fam_pairs = expected_labeled_pairs(
        queries=family_queries,
        q_labels=q_family,
        db_domains=db_domains,
        db_labels=db_family,
    )
    sf_pairs = expected_labeled_pairs(
        queries=superfamily_queries,
        q_labels=q_superfamily,
        db_domains=db_domains,
        db_labels=db_superfamily,
    )
    expected = list(dict.fromkeys(fam_pairs + sf_pairs))
    if path.is_file():
        complete, missing = _cached_pairs_present(path, expected=expected, symmetric=True)
        if complete:
            print(f">>> use cached seq-identity TSV: {path}")
            return path
        print(
            f"[warn] incomplete seq-identity TSV: missing {missing}/"
            f"{len(expected)} expected pairs; recomputing {path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f">>> compute seq-identity cache: {path}")
    print(
        f">>> compute pairs: {len(expected)} "
        f"| family pairs={len(fam_pairs)} | superfamily pairs={len(sf_pairs)} "
        f"| method={SEQ_IDENTITY_METHOD}"
    )
    aligner = make_global_aligner()
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["query_id", "target_id", "seq_identity"])
        for i, (q, t) in enumerate(expected, start=1):
            ident = global_alignment_identity_percent(
                seq_d.get(q, ""), seq_d.get(t, ""), aligner
            )
            writer.writerow([q, t, fmt(ident)])
            if i % 1000 == 0:
                print(f"  seq-identity {i}/{len(expected)}")
    print(f">>> saved seq-identity TSV: {path}")
    return path


def load_seq_identity_bins_by_query(
    path: Path,
    queries: Sequence[str],
    dom_index: Mapping[str, int],
    edges: Sequence[float],
    scale: str,
) -> dict[str, dict[str, str]]:
    qset = {str(q).strip() for q in queries}
    rows: list[tuple[str, str, float]] = []
    max_seen = -math.inf
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        q_col = find_col(reader.fieldnames, ("query_id", "query", "q", "seq_id_1", "id1"))
        t_col = find_col(reader.fieldnames, ("target_id", "target", "t", "seq_id_2", "id2"))
        v_col = find_col(reader.fieldnames, ("seq_identity", "identity", "pident", "seqid"))
        for r in reader:
            q = str(r.get(q_col, "")).strip()
            t = str(r.get(t_col, "")).strip()
            if not q or not t or q not in dom_index or t not in dom_index:
                continue
            if q not in qset and t not in qset:
                continue
            try:
                v = float(str(r.get(v_col, "")).strip())
            except ValueError:
                continue
            if not np.isfinite(v):
                continue
            rows.append((q, t, v))
            max_seen = max(max_seen, v)

    factor = 1.0
    if scale == "fraction" or (scale == "auto" and max_seen <= 1.5):
        factor = 100.0

    out: dict[str, dict[str, str]] = defaultdict(dict)
    for q, t, v in rows:
        lab = value_to_bin(v * factor, edges)
        if lab is None:
            continue
        if q in qset:
            out[q][t] = lab
        if t in qset:
            out[t][q] = lab
    return dict(out)


def _pos_ids_in_topk_for_bin(
    score_row: np.ndarray,
    pos_ids: Sequence[str],
    dom_index: Mapping[str, int],
    top_k: int,
) -> list[str]:
    """NW bin positives that appear in this query's top-K hits (finite score only)."""
    row = np.asarray(score_row, dtype=np.float64)
    n = len(row)
    if n <= 0 or not pos_ids:
        return []
    k_take = min(int(top_k), n)
    order = np.argsort(-row)[:k_take]
    top_idx = {int(j) for j in order if np.isfinite(row[j])}
    return [t for t in pos_ids if int(dom_index[t]) in top_idx]


def collect_binned_topk_retrieval_counts(
    *,
    task: str,
    group_type: str,
    sim_rows: Sequence[np.ndarray],
    queries: Sequence[str],
    q_labels: np.ndarray,
    db_domains: Sequence[str],
    db_labels: np.ndarray,
    dom_index: Mapping[str, int],
    bin_by_query: Mapping[str, Mapping[str, str]],
    bin_labels: Sequence[str],
    top_k_runs: Sequence[int],
) -> list[dict[str, object]]:
    """Per top_k × bin: queries / bin-positive targets found in that model's top-K."""
    rows_out: list[dict[str, object]] = []
    dom_arr = np.asarray(db_domains, dtype=object)

    for top_k in top_k_runs:
        k = int(top_k)
        if k < 1:
            continue
        for bin_lab in bin_labels:
            n_q = 0
            n_pos = 0
            for i, q in enumerate(queries):
                q = str(q).strip()
                q_lab = str(q_labels[i]).strip()
                if not q_lab:
                    continue

                pos_targets = dom_arr[db_labels == q_lab]
                q_bins = bin_by_query.get(q, {})
                pos_ids = [str(t) for t in pos_targets if q_bins.get(str(t)) == bin_lab]
                if not pos_ids:
                    continue

                score_row = np.asarray(sim_rows[i], dtype=np.float32)
                hit = _pos_ids_in_topk_for_bin(score_row, pos_ids, dom_index, k)
                if not hit:
                    continue
                n_q += 1
                n_pos += len(hit)

            rows_out.append(
                {
                    "task": task,
                    "group_type": group_type,
                    "top_k": k,
                    "group_name": bin_lab,
                    "num_queries": n_q,
                    "num_positive_pairs": n_pos,
                }
            )
    return rows_out


def collect_binned_retrieve_metrics(
    *,
    task: str,
    group_type: str,
    sim_rows: Sequence[np.ndarray],
    queries: Sequence[str],
    q_labels: np.ndarray,
    q_soft_labels: np.ndarray | None,
    db_domains: Sequence[str],
    db_labels: np.ndarray,
    db_soft_labels: np.ndarray | None,
    dom_index: Mapping[str, int],
    bin_by_query: Mapping[str, Mapping[str, str]],
    bin_labels: Sequence[str],
) -> list[dict[str, object]]:
    rows_out: list[dict[str, object]] = []
    dom_arr = np.asarray(db_domains, dtype=object)
    arange_dom = np.arange(len(db_domains), dtype=np.int64)

    for bin_lab in bin_labels:
        aucs: list[float] = []
        auc_qids: list[str] = []
        n_q = 0
        n_pos_total = 0
        n_missing_positive = 0
        n_positive_seen = 0
        n_neg_total = 0

        for i, q in enumerate(queries):
            q = str(q).strip()
            q_lab = str(q_labels[i]).strip()
            q_soft = "" if q_soft_labels is None else str(q_soft_labels[i]).strip()
            if q_soft.lower() == "nan":
                q_soft = ""
            if not q_lab:
                continue

            pos_label_mask = db_labels == q_lab
            if q_soft and db_soft_labels is not None:
                neg_mask = (db_labels != q_lab) & (db_soft_labels != q_soft)
            else:
                neg_mask = db_labels != q_lab
            pos_targets = dom_arr[pos_label_mask]
            n_positive_seen += int(pos_label_mask.sum())

            q_bins = bin_by_query.get(q, {})
            pos_ids = [str(t) for t in pos_targets if q_bins.get(str(t)) == bin_lab]
            n_missing_positive += sum(1 for t in pos_targets if str(t) not in q_bins)
            if not pos_ids:
                continue

            score_row = np.asarray(sim_rows[i], dtype=np.float32)
            pos_idx = np.fromiter((dom_index[t] for t in pos_ids), dtype=np.int64)

            n_q += 1
            n_pos_total += len(pos_ids)
            n_neg_total += int(neg_mask.sum())

            pos = np.zeros(len(db_domains), dtype=bool)
            pos[pos_idx] = True
            if q_soft and db_soft_labels is not None:
                fp_eligible = (
                    (db_labels != q_lab)
                    & (db_soft_labels != q_soft)
                    & (arange_dom != int(dom_index[q]))
                )
            else:
                fp_eligible = (db_labels != q_lab) & (arange_dom != int(dom_index[q]))
            if int(fp_eligible.sum()) <= 0:
                aucs.append(1.0)
                auc_qids.append(q)
                continue

            order = np.lexsort((arange_dom, -score_row))
            tp_before = 0
            p = int(pos.sum())
            auc = 1.0
            for j in order:
                jj = int(j)
                if fp_eligible[jj]:
                    auc = float(tp_before) / float(p)
                    break
                if pos[jj]:
                    tp_before += 1
            aucs.append(auc)
            auc_qids.append(q)

        auc_arr = np.asarray(aucs, dtype=np.float64)
        finite = np.isfinite(auc_arr)
        rows_out.append(
            {
                "task": task,
                "group_type": group_type,
                "group_name": bin_lab,
                "num_queries": n_q,
                "num_positive_pairs": n_pos_total,
                "num_negative_pairs": n_neg_total,
                "num_label_positive_pairs_seen": n_positive_seen,
                "num_label_positive_pairs_missing_bin": n_missing_positive,
                "AUC-1FP": (
                    float(np.mean(auc_arr[finite])) if finite.any() else float("nan")
                ),
                "per_query_aucs": aucs,
                "per_query_ids": auc_qids,
            }
        )

    return rows_out


def prepare_multibin_inputs(
    *,
    seq_identity_tsv: Path,
    family_queries: Sequence[str],
    q_family: np.ndarray,
    superfamily_queries: Sequence[str],
    q_superfamily: np.ndarray,
    db_domains: Sequence[str],
    db_family: np.ndarray,
    db_superfamily: np.ndarray,
    seq_d: Mapping[str, str],
    dom_index: Mapping[str, int],
    seq_identity_edges: Sequence[float],
    seq_identity_scale: str,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    seq_identity_tsv = ensure_seq_identity_tsv_for_benchmark_queries(
        seq_identity_tsv,
        family_queries=family_queries,
        q_family=q_family,
        superfamily_queries=superfamily_queries,
        q_superfamily=q_superfamily,
        db_domains=db_domains,
        db_family=db_family,
        db_superfamily=db_superfamily,
        seq_d=seq_d,
    )

    all_queries = list(
        dict.fromkeys([str(q) for q in family_queries] + [str(q) for q in superfamily_queries])
    )
    seq_bins = load_seq_identity_bins_by_query(
        seq_identity_tsv,
        all_queries,
        dom_index,
        seq_identity_edges,
        seq_identity_scale,
    )
    seq_labels = [bin_label(seq_identity_edges, i) for i in range(len(seq_identity_edges) - 1)]
    return seq_bins, seq_labels


def collect_multibin_rows(
    *,
    family_rows: Sequence[np.ndarray] | None,
    family_queries: Sequence[str],
    q_family: np.ndarray,
    superfamily_rows: Sequence[np.ndarray] | None,
    superfamily_queries: Sequence[str],
    q_superfamily: np.ndarray,
    db_domains: Sequence[str],
    db_family: np.ndarray,
    db_superfamily: np.ndarray,
    db_fold: np.ndarray,
    dom_index: Mapping[str, int],
    seq_bins: Mapping[str, Mapping[str, str]],
    seq_labels: Sequence[str],
) -> list[dict[str, object]]:
    q_fam_superfamily = np.array(
        [db_superfamily[dom_index[q]] for q in family_queries], dtype=object
    )
    q_sf_fold = np.array([db_fold[dom_index[q]] for q in superfamily_queries], dtype=object)

    def _metric_rows(
        group_type: str,
        bins: Mapping[str, Mapping[str, str]],
        labels: Sequence[str],
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        if family_rows is not None:
            rows.extend(
                collect_binned_retrieve_metrics(
                    task="family",
                    group_type=group_type,
                    sim_rows=family_rows,
                    queries=family_queries,
                    q_labels=q_family,
                    q_soft_labels=q_fam_superfamily,
                    db_domains=db_domains,
                    db_labels=db_family,
                    db_soft_labels=db_superfamily,
                    dom_index=dom_index,
                    bin_by_query=bins,
                    bin_labels=labels,
                )
            )
        if superfamily_rows is not None:
            rows.extend(
                collect_binned_retrieve_metrics(
                    task="superfamily",
                    group_type=group_type,
                    sim_rows=superfamily_rows,
                    queries=superfamily_queries,
                    q_labels=q_superfamily,
                    q_soft_labels=q_sf_fold,
                    db_domains=db_domains,
                    db_labels=db_superfamily,
                    db_soft_labels=db_fold,
                    dom_index=dom_index,
                    bin_by_query=bins,
                    bin_labels=labels,
                )
            )
        return rows

    return _metric_rows("seq_identity", seq_bins, seq_labels)


def collect_multibin_topk_count_rows(
    *,
    family_rows: Sequence[np.ndarray] | None,
    family_queries: Sequence[str],
    q_family: np.ndarray,
    superfamily_rows: Sequence[np.ndarray] | None,
    superfamily_queries: Sequence[str],
    q_superfamily: np.ndarray,
    db_domains: Sequence[str],
    db_family: np.ndarray,
    db_superfamily: np.ndarray,
    dom_index: Mapping[str, int],
    seq_bins: Mapping[str, Mapping[str, str]],
    seq_labels: Sequence[str],
    top_k_runs: Sequence[int] | None = None,
) -> list[dict[str, object]]:
    if top_k_runs is None:
        top_k_runs = DEFAULT_MULTIBIN_TOP_K_RUNS

    def _count_rows(
        group_type: str,
        bins: Mapping[str, Mapping[str, str]],
        labels: Sequence[str],
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        if family_rows is not None:
            rows.extend(
                collect_binned_topk_retrieval_counts(
                    task="family",
                    group_type=group_type,
                    sim_rows=family_rows,
                    queries=family_queries,
                    q_labels=q_family,
                    db_domains=db_domains,
                    db_labels=db_family,
                    dom_index=dom_index,
                    bin_by_query=bins,
                    bin_labels=labels,
                    top_k_runs=top_k_runs,
                )
            )
        if superfamily_rows is not None:
            rows.extend(
                collect_binned_topk_retrieval_counts(
                    task="superfamily",
                    group_type=group_type,
                    sim_rows=superfamily_rows,
                    queries=superfamily_queries,
                    q_labels=q_superfamily,
                    db_domains=db_domains,
                    db_labels=db_superfamily,
                    dom_index=dom_index,
                    bin_by_query=bins,
                    bin_labels=labels,
                    top_k_runs=top_k_runs,
                )
            )
        return rows

    return _count_rows("seq_identity", seq_bins, seq_labels)


def _load_or_create_multibin_workbook(path: Path):
    try:
        from openpyxl import Workbook, load_workbook
    except ImportError as e:
        raise RuntimeError(
            "Writing multibin Excel summaries requires openpyxl. "
            "Install it in this environment or use an environment that includes it."
        ) from e

    if path.is_file():
        wb = load_workbook(path)
    else:
        wb = Workbook()
        default = wb.active
        wb.remove(default)
    for sheet_name in MULTIBIN_EXCEL_SHEETS:
        if sheet_name not in wb.sheetnames:
            ws = wb.create_sheet(sheet_name)
            ws.append(["Model"])
    return wb


def _remove_rows_with_first_col(ws, value: str) -> None:
    value_str = str(value).strip()
    if ws.max_row is None or ws.max_row < 2:
        return
    for row_idx in range(ws.max_row, 1, -1):
        cell_val = ws.cell(row=row_idx, column=1).value
        if cell_val is not None and str(cell_val).strip() == value_str:
            ws.delete_rows(row_idx, 1)


def _int_count(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _ordered_multibin_count_rows(
    method: str,
    count_rows: Sequence[Mapping[str, object]],
) -> list[tuple[str, str, int, str, int, int]]:
    """Per model/task/top_k/bin: queries and bin-positive targets in top-K."""
    by_task: dict[str, list[Mapping[str, object]]] = {t: [] for t in MULTIBIN_COUNT_TASK_ORDER}
    for row in count_rows:
        task = str(row.get("task", "")).strip()
        if task in by_task:
            by_task[task].append(row)

    out: list[tuple[str, str, int, str, int, int]] = []
    for task in MULTIBIN_COUNT_TASK_ORDER:
        task_rows = by_task[task]
        task_rows.sort(key=lambda r: (_int_count(r.get("top_k")), str(r.get("group_name", ""))))
        for row in task_rows:
            out.append(
                (
                    method,
                    task,
                    _int_count(row.get("top_k")),
                    bin_display_label(str(row.get("group_name", ""))),
                    _int_count(row.get("num_queries")),
                    _int_count(row.get("num_positive_pairs")),
                )
            )
    return out


def print_multibin_counts(
    method: str,
    count_rows: Sequence[Mapping[str, object]],
) -> None:
    """Log per top_k × bin query/target counts (bin positives hit in model top-K)."""
    rows = _ordered_multibin_count_rows(method, count_rows)
    if not rows:
        print(f">>> multibin counts [{method}]: (no rows)")
        return
    print(
        f">>> multibin counts [{method}] "
        f"(bin positives found in model top-K; per task/top_k/bin):"
    )
    current_task = ""
    current_k = -1
    for _method, task, top_k, bin_lab, n_q, n_pairs in rows:
        if task != current_task:
            current_task = task
            current_k = -1
            print(f"  [{task}]")
        if top_k != current_k:
            current_k = top_k
            print(f"    top_{top_k}")
        print(f"      {bin_lab}\tn_queries={n_q}\tn_positive_pairs={n_pairs}")


def _remove_multibin_count_rows_for_model(ws, method: str) -> None:
    method_str = str(method).strip()
    if ws.max_row is None or ws.max_row < 2:
        return
    for row_idx in range(ws.max_row, 1, -1):
        cell_val = ws.cell(row=row_idx, column=1).value
        if cell_val is not None and str(cell_val).strip() == method_str:
            ws.delete_rows(row_idx, 1)


def _write_multibin_counts_sheet(
    wb,
    method: str,
    count_rows: Sequence[Mapping[str, object]],
) -> None:
    """Append per-model rows: bin positives retrieved in top-K for each task/bin."""
    if MULTIBIN_COUNTS_SHEET not in wb.sheetnames:
        ws = wb.create_sheet(MULTIBIN_COUNTS_SHEET, 0)
        ws.append(list(MULTIBIN_COUNTS_HEADER))
    else:
        ws = wb[MULTIBIN_COUNTS_SHEET]
        header = [cell.value for cell in ws[1]]
        if header != list(MULTIBIN_COUNTS_HEADER):
            wb.remove(ws)
            ws = wb.create_sheet(MULTIBIN_COUNTS_SHEET, 0)
            ws.append(list(MULTIBIN_COUNTS_HEADER))
    _remove_multibin_count_rows_for_model(ws, method)
    for row in _ordered_multibin_count_rows(method, count_rows):
        ws.append(list(row))


def _append_multibin_sheet_row(wb, sheet_name: str, method: str, rows: Sequence[Mapping[str, object]]) -> None:
    ws = wb[sheet_name]
    _remove_rows_with_first_col(ws, method)
    values_by_bin: dict[str, str] = {}
    for r in rows:
        lab = bin_display_label(str(r.get("group_name", "")))
        try:
            auc = float(r.get("AUC-1FP", float("nan")))
        except (TypeError, ValueError):
            auc = float("nan")
        values_by_bin[lab] = fmt(auc)

    header = [cell.value for cell in ws[1]]
    for lab in values_by_bin:
        if lab not in header:
            ws.cell(row=1, column=len(header) + 1, value=lab)
            header.append(lab)

    row = [""] * len(header)
    row[0] = method
    for lab, value in values_by_bin.items():
        row[header.index(lab)] = value
    ws.append(row)


def _get_or_reset_sheet(wb, sheet_name: str, header: Sequence[object]):
    """Return a sheet with the exact given header, recreating it if the header differs."""
    if sheet_name not in wb.sheetnames:
        ws = wb.create_sheet(sheet_name)
        ws.append(list(header))
        return ws
    ws = wb[sheet_name]
    existing = [cell.value for cell in ws[1]]
    if existing != list(header):
        wb.remove(ws)
        ws = wb.create_sheet(sheet_name)
        ws.append(list(header))
    return ws


def _write_seq_auc_distribution_sheet(
    wb,
    method: str,
    seq_rows: Sequence[Mapping[str, object]],
) -> None:
    """Long format: one row per (model, task, bin, query) with that query's AUC-1FP."""
    ws = _get_or_reset_sheet(wb, SEQ_AUC_DIST_SHEET, SEQ_AUC_DIST_HEADER)
    _remove_rows_with_first_col(ws, method)
    for task in MULTIBIN_COUNT_TASK_ORDER:
        for r in seq_rows:
            if str(r.get("task", "")) != task:
                continue
            bin_lab = bin_display_label(str(r.get("group_name", "")))
            qids = list(r.get("per_query_ids", []) or [])
            aucs = list(r.get("per_query_aucs", []) or [])
            for qid, auc in zip(qids, aucs):
                ws.append([method, task, bin_lab, str(qid), fmt(float(auc))])


def _write_seq_auc_stats_sheet(
    wb,
    method: str,
    seq_rows: Sequence[Mapping[str, object]],
) -> None:
    """Per (model, task, bin) summary of the query AUC-1FP distribution."""
    ws = _get_or_reset_sheet(wb, SEQ_AUC_STATS_SHEET, SEQ_AUC_STATS_HEADER)
    _remove_rows_with_first_col(ws, method)
    for task in MULTIBIN_COUNT_TASK_ORDER:
        for r in seq_rows:
            if str(r.get("task", "")) != task:
                continue
            bin_lab = bin_display_label(str(r.get("group_name", "")))
            arr = np.asarray(list(r.get("per_query_aucs", []) or []), dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            n = int(arr.size)
            if n == 0:
                ws.append([method, task, bin_lab, 0, "", "", "", "", "", "", "", ""])
                continue
            std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            var = float(np.var(arr, ddof=1)) if n > 1 else 0.0
            ws.append(
                [
                    method,
                    task,
                    bin_lab,
                    n,
                    fmt(float(np.mean(arr))),
                    fmt(std),
                    fmt(var),
                    fmt(float(arr.min())),
                    fmt(float(np.percentile(arr, 25))),
                    fmt(float(np.percentile(arr, 50))),
                    fmt(float(np.percentile(arr, 75))),
                    fmt(float(arr.max())),
                ]
            )


def write_multibin_excel_outputs(
    *,
    method: str,
    metrics_xlsx: Path,
    seq_rows: Sequence[Mapping[str, object]],
    count_rows: Sequence[Mapping[str, object]],
) -> None:
    metrics_xlsx = metrics_xlsx.with_suffix(".xlsx")
    metrics_xlsx.parent.mkdir(parents=True, exist_ok=True)
    wb = _load_or_create_multibin_workbook(metrics_xlsx)

    _append_multibin_sheet_row(
        wb,
        "sequence_family",
        method,
        [r for r in seq_rows if str(r.get("task", "")) == "family"],
    )
    _append_multibin_sheet_row(
        wb,
        "sequence_superfamily",
        method,
        [r for r in seq_rows if str(r.get("task", "")) == "superfamily"],
    )
    _write_multibin_counts_sheet(wb, method, count_rows)
    _write_seq_auc_stats_sheet(wb, method, seq_rows)
    _write_seq_auc_distribution_sheet(wb, method, seq_rows)
    wb.save(metrics_xlsx)
    print_multibin_counts(method, count_rows)
