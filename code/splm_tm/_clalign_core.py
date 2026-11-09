"""Vendored subset of CLAlign: the global alignment kernel and its F1 metric.

Third-party code, copied so that candidate alignment works from this repository
alone. CLAlign remains preferred when it is installed: ``splm_tm.alignment``
imports the upstream package first and falls back to this module only when that
import fails, so a machine with CLAlign keeps the exact code path that produced
the published numbers.

Source: https://github.com/yourh/CLAlign — ``src/clalign/alignment.py`` and
``src/clalign/metrics.py``.

Copied verbatim: ``ProteinSeq``, ``AlignmentResult``, ``get_coverage``,
``align_core`` and ``f1score``.

Modifications, all of them removals:
  * ``pairs_align`` / ``pairwise_align`` (the ``parallel=True`` batch drivers),
    ``draw_alignment``, ``empty_protein_seq`` and ``empty_aln_res`` are not
    included; nothing here calls them. ``align_core`` itself is serial
    ``njit``, so this path does not depend on a numba threading layer and the
    upstream ``__init__`` setting ``THREADING_LAYER = 'tbb'`` is not reproduced.
  * ``ProteinSeq.get_record`` is not included, which drops the Biopython
    import it needed.
  * ``hec_acc`` and ``hec_sov`` from ``metrics.py`` are not included.

No line of the retained code was altered. Requires numba, which the
repository's ``requirements.txt`` installs; ``pip install -e ".[align]"`` covers
it when installing the package alone.

----------------------------------------------------------------------------
MIT License

Copyright (c) 2026 Ronghui You

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
----------------------------------------------------------------------------
"""
import numpy as np
import numba

__all__ = ["ProteinSeq", "AlignmentResult", "align_core", "f1score"]


@numba.experimental.jitclass({
    'seq': numba.types.unicode_type,
    'pid': numba.types.unicode_type,
    'desc': numba.types.unicode_type,
    'idx': numba.i4
})
class ProteinSeq(object):
    """

    """
    def __init__(self, seq, pid='', desc='', idx=0):
        self.seq, self.pid, self.desc, self.idx = seq.upper(), pid, desc, idx

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, item):
        return self.seq[item]


@numba.experimental.jitclass({
    'seq1': numba.extending.as_numba_type(ProteinSeq),
    'seq2': numba.extending.as_numba_type(ProteinSeq),
    'aln1': numba.types.unicode_type,
    'aln2': numba.types.unicode_type,
    'start1': numba.i4,
    'start2': numba.i4,
    'score': numba.f4,
    'coverage': numba.f4
})
class AlignmentResult(object):
    """

    """

    def __init__(self, seq1, seq2, aln1, aln2=None, start1=0, start2=0, score=0.0, coverage=0.0):
        if aln2 is not None:
            assert len(aln1) == len(aln2)
        else:
            i, j, aln1, aln2, aln_text = 0, 0, '', '', aln1
            for k in range(len(aln_text)):
                match aln_text[k]:
                    case ':' | '.': aln1, aln2, i, j = aln1 + seq1[i], aln2 + seq2[j], i + 1, j + 1
                    case '1': aln1, aln2, i = aln1 + seq1[i], aln2 + '-', i + 1
                    case '2': aln1, aln2, j = aln1 + '-', aln2 + seq2[j], j + 1
                    case _: raise ValueError
            assert i == len(seq1) and j == len(seq2)
        self.seq1, self.seq2, self.aln1, self.aln2, self.start1, self.start2 = seq1, seq2, aln1, aln2, start1, start2
        self.score, self.coverage = score, coverage
        for x, y in zip(self.aln1, self.aln2):
            assert x != '-' or y != '-'
    
    def __lt__(self, other):
        return self.score < other.score

    def __len__(self):
        return len(self.aln1)

    def get_matched_res(self):
        i, j, pos = self.start1, self.start2, []
        for k in range(len(self.aln1)):
            if self.aln1[k] != '-' and self.aln2[k] != '-':
                pos.append((i, j))
            i += self.aln1[k] != '-'
            j += self.aln2[k] != '-'
        return pos


@numba.njit(nogil=True, fastmath=True)
def get_coverage(aln1, aln2, seq_len):
    return len([k for k in range(len(aln1)) if aln1[k] != '-' and aln2[k] != '-']) / seq_len


@numba.njit(nogil=True, fastmath=True)
def align_core(seq1: ProteinSeq, seq2: ProteinSeq, sub_mat, gap_penalty=0.0, local=False,
               only_score=False) -> AlignmentResult:
    assert (len(seq1), len(seq2)) == sub_mat.shape
    assert gap_penalty >= 0.0
    f = np.empty((sub_mat.shape[0] + 1, sub_mat.shape[1] + 1), dtype=np.float32)
    g = np.empty_like(f, dtype=np.int32)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            x = np.asarray((0.0 if local or i == j == 0 else -np.inf,
                            f[i - 1, j - 1] + sub_mat[i - 1, j - 1] if i > 0 and j > 0 else -np.inf,
                            f[i - 1, j] - gap_penalty if i > 0 else -np.inf,
                            f[i, j - 1] - gap_penalty if j > 0 else -np.inf))
            g[i, j], f[i, j] = (y := x.argmax()), x[y]
    if local:
        x, y, m = -1, -1, -np.inf
        for i in range(f.shape[0]):
            for j in range(f.shape[1]):
                if f[i, j] > m:
                    m, x, y = f[i, j], i, j
    else:
        x, y = sub_mat.shape
    (i, j), aln1, aln2, s = (x, y), '', '', 0.0
    while i > 0 or j > 0:
        match g[i, j]:
            case 1: aln1, aln2, s = seq1[i := i - 1] + aln1, seq2[j := j - 1] + aln2, s + sub_mat[i, j]
            case 2: aln1, aln2 = seq1[i := i - 1] + aln1, '-' + aln2
            case 3: aln1, aln2 = '-' + aln1, seq2[j := j - 1] + aln2
            case 0: break
    return AlignmentResult(seq1, seq2, aln1 if not only_score else '', aln2 if not only_score else '', i, j,
                           s / (0.5 * (len(seq1) + len(seq2))),
                           get_coverage(aln1, aln2, len(seq1)))


def f1score(truth: AlignmentResult, pred: AlignmentResult):
    t_, p_ = set(truth.get_matched_res()), set(pred.get_matched_res())
    pr_, re_ = (x_ := len(t_ & p_)) / (len(p_) + 1e-10), x_ / (len(t_) + 1e-10)
    return pr_, re_, 2 * pr_ * re_ / (pr_ + re_ + 1e-10)
