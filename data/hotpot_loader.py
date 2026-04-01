import json

_data_map = None

def load_hotpotqa_qa(file_path="/home/tien/my_project/multiQ/data/hotpot500.json"):
    global _data_map

    if _data_map is None:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        _data_map = {
            item["_id"]: {
                "question": item["question"],
                "answer": item["answer"],
                "context": item["context"]
            }
            for item in data
        }

    return _data_map