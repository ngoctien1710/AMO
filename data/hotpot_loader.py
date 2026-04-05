import json

_data_map = None

<<<<<<< HEAD
def load_hotpotqa_qa(file_path="/home/tien/my_project/AMO/data/hotpot500.json"):
=======
def load_hotpotqa_qa(file_path="/home/tientn/TIEN/best/AMO/data/hotpot500.json"):
>>>>>>> a540a6f (EM 30.8, F1 44.13)
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