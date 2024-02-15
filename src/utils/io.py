from typing import Dict
import os
import json


def save_json(data: object, json_path: str) -> None:
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json(json_path: str) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
