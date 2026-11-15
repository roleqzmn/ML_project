import math
import numpy as np


def checked(y, scores):
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or scores.shape != y.shape or not len(y):
        raise ValueError("Expected nonempty, equally sized 1D labels and scores")
    if set(y.tolist()) != {0, 1}:
        raise ValueError("Both classes (0 and 1) must be present")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite and between 0 and 1")
    return y, scores


def metrics(y, scores, threshold):
    y, scores = checked(y, scores)
    accept = scores >= threshold
    tp = int(np.sum(accept & (y == 1)))
    fp = int(np.sum(accept & (y == 0)))
    fn = int(np.sum(~accept & (y == 1)))
    tn = int(np.sum(~accept & (y == 0)))
    f1_allowed = 2 * tp / max(2 * tp + fp + fn, 1)
    f1_denied = 2 * tn / max(2 * tn + fp + fn, 1)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(y),
        "macro_f1": (f1_allowed + f1_denied) / 2,
        "balanced_accuracy": (tp / (tp + fn) + tn / (tn + fp)) / 2,
        "FAR": fp / (fp + tn),
        "FRR": fn / (fn + tp),
        "allowed_recall": tp / (tp + fn),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def best_f1_threshold(y, scores):
    y, scores = checked(y, scores)
    order = np.argsort(-scores, kind="stable")
    s, labels = scores[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(labels)[ends]
    fp = ends + 1 - tp
    fn, tn = y.sum() - tp, (y == 0).sum() - fp
    f1 = (2 * tp / np.maximum(2 * tp + fp + fn, 1)
          + 2 * tn / np.maximum(2 * tn + fp + fn, 1)) / 2
    reject_all = metrics(y, scores, math.nextafter(float(s[0]), math.inf))
    values = np.r_[reject_all["macro_f1"], f1]
    thresholds = np.r_[reject_all["threshold"], s[ends]]
    return float(thresholds[int(np.argmax(values))])


def far_threshold(y, scores, target_far):
    y, scores = checked(y, scores)
    if not 0 <= target_far < 1:
        raise ValueError("target_far must be in [0, 1)")
    negative = np.sort(scores[y == 0])[::-1]
    budget = math.floor(target_far * len(negative))
    return math.nextafter(float(negative[budget]), math.inf)
