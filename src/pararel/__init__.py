import json
import os
from typing import Dict
import urllib.request
from pathlib import Path

from tqdm import tqdm


def get_pararel_data(pararel_dir: str) -> Dict:
    data = dict()

    raw_data_dir = os.path.join(pararel_dir, "pararel")

    relation_files = os.listdir(os.path.join(raw_data_dir, "relations"))
    subjects_objects_files = os.listdir(os.path.join(raw_data_dir, "subjects_objects"))

    for file in tqdm(relation_files):
        with open(os.path.join(raw_data_dir, "relations", file)) as f:
            content = f.readlines()
        r = os.path.splitext(os.path.basename(file))[0]
        data[r] = dict()
        pararel_templates = [json.loads(d.strip()) for d in content if d]
        relations = [
            {
                "template": template["pattern"].replace("[X]", "{subject}").replace("[Y]", "{object}"),
                "lemma": template["extended_lemma"],
            }
            for template in pararel_templates
        ]
        data[r]["relations"] = relations
    for file in tqdm(subjects_objects_files):
        with open(os.path.join(raw_data_dir, "subjects_objects", file)) as f:
            content = f.readlines()
        r = os.path.splitext(os.path.basename(file))[0]
        vocab = [json.loads(d.strip()) for d in content if d]
        facts = [{"subject": entry["sub_label"], "object": entry["obj_label"]} for entry in vocab]
        data[r]["subjects_objects"] = facts

    return data
