"""Float32 scores with a feature-order reduction independent of block shape.

Each product and addition rounds separately. No BLAS reduction or fused multiply
add is used, so a dense call and any query/target partition have identical bits
on the same NumPy platform. Historical benchmark code deliberately stays separate.
"""
import numpy as np


def squared_norms(values):
    result = np.zeros(len(values), dtype=np.float32)
    temporary = np.empty_like(result)
    for feature in range(values.shape[1]):
        np.multiply(values[:, feature], values[:, feature], out=temporary)
        np.add(result, temporary, out=result)
    return result


def fixed_scores(queries, targets, *, negative_l2=False):
    """Return a dense block using the same per-pair operations for every shape."""
    scores = np.zeros((len(queries), len(targets)), dtype=np.float32)
    temporary = np.empty_like(scores)
    for feature in range(queries.shape[1]):
        left, right = queries[:, feature, None], targets[None, :, feature]
        if negative_l2:
            np.subtract(left, right, out=temporary)
            np.multiply(temporary, temporary, out=temporary)
        else:
            np.multiply(left, right, out=temporary)
        np.add(scores, temporary, out=scores)
    if negative_l2:
        np.sqrt(scores, out=scores)
        np.negative(scores, out=scores)
    return scores
