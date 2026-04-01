# evaluate.py
from evaluation.metrics import exact_match_score, compute_precision_recall_f1

def evaluate(gt, pred):
    """
    Args:
        gt: str (ground truth)
        pred: str (prediction)
    Returns:
        em: int (0 hoặc 1)
        precision: float
        recall: float
        f1: float
    """
    em = exact_match_score(pred, gt)
    precision, recall, f1 = compute_precision_recall_f1(pred, gt)
    return em, precision, recall, f1