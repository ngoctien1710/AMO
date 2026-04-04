from model.custom_language_model import LlamaLLM
from methods.AMO import AMO
from data.hotpot_loader import load_hotpotqa_qa
from evaluation.evaluate import evaluate
import os
from tqdm import tqdm
import torch
from utils.utils import save_predictions, save_metrics

def main():

    data = load_hotpotqa_qa()
    model = LlamaLLM()
    method = AMO(model = model)

    total_em = 0
    total_f1 = 0
    total_precision = 0
    total_recall = 0
    max_steps = 20
    results = []
    for _, (qid, qa) in enumerate(
        tqdm(data.items(), total=len(data), desc=f"Running {method.name} on HotpotQA")
    ):
        if max_steps <= 0:
            break
        max_steps -= 1
        question = qa["question"]
        gt_answer = qa["answer"]

        try:
            run_history = method.inference(question, qid=qid)

            if run_history:
                last_step = max(run_history.keys())
                final_action = run_history[last_step].get("final") or ""
            else:
                final_action = ""
        except Exception as e:
            run_history = {}
            print(f"Error at sample {qid}: {e}")
            final_action = ""
        em, precision, recall, f1 = evaluate(gt_answer, final_action)
        total_em += em
        total_f1 += f1
        total_precision += precision
        total_recall += recall

        results.append({
            "id": qid,
            "question": question,
            "ground_truth": gt_answer,
            "history": run_history,
            "prediction": final_action,
            "em": em,
            "f1": f1,
            "precision": precision,
            "recall": recall
        })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    n = len(results)

    avg_em = total_em / n
    avg_f1 = total_f1 / n
    avg_recall = total_recall / n
    avg_precision = total_precision / n

    print("=================================")
    print("Average EM:", avg_em)
    print("=================================")

    os.makedirs(f"results/{method.name}", exist_ok=True)
    
    avg_calls = method.model.get_call_count() / n
    # save metrics
    metrics = {
        "method": method.name,
        "samples": n,
        "average_em": avg_em,
        "avg_llm_calls": avg_calls,
        "avg_f1" : avg_f1,
        "avg_recall" : avg_recall,
        "avg_precision" : avg_precision
    }

    save_predictions(f"results/{method.name}/{method.name}_predictions.json", results)
    save_metrics(f"results/{method.name}/{method.name}_metrics.json", metrics)

if __name__ == "__main__":
    main()