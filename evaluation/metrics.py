import re
import string
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def white_space_fix(text):
        return ' '.join(text.split())
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    # 🔥 return int để tính trung bình EM
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_precision_recall_f1(prediction, ground_truth):
    """
    Compute token-level precision, recall, and F1 between prediction and ground truth
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    # handle empty prediction / ground truth
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        p = 1.0 if pred_tokens == gt_tokens else 0.0
        r = p
        f1 = p
        return p, r, f1

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1