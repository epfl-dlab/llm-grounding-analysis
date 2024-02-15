from collections import defaultdict
import copy
import json
import os
from typing import Dict, List, Tuple
import numpy as np
import torch

from tqdm import tqdm
from src.model import adapt_target_tokens, get_next_token_probabilities
from src.utils.logger import get_logger

import torch.nn as nn
from tokenizers import Tokenizer


def extract_data(raw_data_dir: str) -> Dict:
    logger = get_logger()
    relation_files = os.listdir(os.path.join(raw_data_dir, "relations"))
    subjects_objects_files = os.listdir(os.path.join(raw_data_dir, "subjects_objects"))
    logger.info(f"Found {len(relation_files)} relation files and {len(subjects_objects_files)} subjects_objects files")

    data = dict()

    for file in tqdm(relation_files, desc="Extracting ParaRel relations"):
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

    discarded_facts = 0
    total_facts = 0

    for file in tqdm(subjects_objects_files, desc="Extracting ParRel subjects and objects"):
        with open(os.path.join(raw_data_dir, "subjects_objects", file)) as f:
            content = f.readlines()
        r = os.path.splitext(os.path.basename(file))[0]
        vocab = [json.loads(d.strip()) for d in content if d]
        facts = defaultdict(set)
        for entry in vocab:
            if entry["obj_label"] in facts[entry["sub_label"]]:
                discarded_facts += 1
            total_facts += 1
            facts[entry["sub_label"]].add(entry["obj_label"])

        # Convert the dictionary to a list of facts
        facts_list = []
        for sub in facts:
            for obj in facts[sub]:
                facts_list.append({"subject": sub, "object": obj})

        data[r]["subjects_objects"] = facts_list

    logger.info(f"Discarded {discarded_facts} facts out of {total_facts} facts.")
    logger.info(f"Extracted {total_facts-discarded_facts} facts from ParaRel.")

    return data


def choose_false_objects(
    entry, rel_lemma_to_objs, num_false_objects: int, model: nn.Module, tokenizer: Tokenizer
) -> List[str]:
    # Retrieve suitable alt objects
    objs = copy.deepcopy(rel_lemma_to_objs[entry["rel_p_id"]][entry["rel_lemma"]])

    # Remove objects that should not be considered
    objs.remove(entry["object"])

    # If there are no objs with the same lemma, use all the objects with the same relation id to retrieve more objects
    if len(objs) < num_false_objects:
        objs_extended = set()
        for relation_lemma_key in rel_lemma_to_objs[entry["rel_p_id"]].keys():
            objs_extended = objs_extended.union(rel_lemma_to_objs[entry["rel_p_id"]][relation_lemma_key])
        objs_extended = set(list(objs_extended)[: (num_false_objects - len(objs))])
        objs = objs.union(objs_extended)

    assert len(objs) >= num_false_objects, "No alternate objects found"

    # Convert to list
    objs = list(objs)

    # For each object, compute the probability of the query
    target_tokens = adapt_target_tokens(tokenizer, objs, preprend_space=True)[0]

    # Run the model and check their probabilities
    probs = get_next_token_probabilities(
        model=model,
        tokenizer=tokenizer,
        prompts=entry["query"],
        target_tokens=target_tokens,
        device="cuda",
    )
    probs = probs.squeeze(0)

    # Choose the num_false_objects objects with the lowest probability
    alt_objs = [objs[i] for i in torch.argsort(probs)[:num_false_objects]]

    return alt_objs


def choose_best_template(entry, model: nn.Module, tokenizer: Tokenizer) -> Tuple[str, str, float, int]:
    queries = [
        relation["template"].replace("{subject}", entry["subject"]).split("{object}")[0].strip()
        for relation in entry["relations"]
    ]
    target_tokens = adapt_target_tokens(tokenizer, [entry["object"]], preprend_space=True)[0]

    probs = []
    ranks = []
    for query in queries:
        prob, rank = get_next_token_probabilities(
            model=model,
            tokenizer=tokenizer,
            prompts=query,
            target_tokens=target_tokens,
            device="cuda",
        )
        prob, rank = prob.squeeze(0)[0].item(), rank.squeeze(0)[0].item()
        probs.append(prob)
        ranks.append(rank)
    argmax = np.argmax(probs)

    return queries[argmax], entry["relations"][argmax]["lemma"], probs[argmax], ranks[argmax]


def process_raw_data(raw_data, model: nn.Module, tokenizer: Tokenizer, num_false_objects: int) -> List[Dict]:
    entries = []
    for rel_p_id in raw_data.keys():
        for entry in raw_data[rel_p_id]["subjects_objects"]:
            entry["relations"] = raw_data[rel_p_id]["relations"]
            entry["rel_p_id"] = rel_p_id
            entries.append(entry)

    for entry in tqdm(entries, desc="Choosing best templates"):
        entry["query"], entry["rel_lemma"], entry["prob"], entry["rank"] = choose_best_template(entry, model, tokenizer)

    rel_lemma_to_objs = defaultdict(lambda: defaultdict(set))
    for entry in tqdm(entries, desc="Grouping objects by relation and relation lemma"):
        rel_lemma_to_objs[entry["rel_p_id"]][entry["rel_lemma"]].add(entry["object"])

    # Filter only the entries where the model gives rank 1 to the correct answer
    entries = [entry for entry in entries if entry["rank"] == 1]

    # Sort the entries by probability
    entries = sorted(entries, key=lambda x: -x["prob"])

    logger = get_logger()

    # For each entry, choose num_false_objects false objects
    extended_pararel = []
    for entry in tqdm(entries, desc="Choosing false objects"):
        entry["false_objects"] = choose_false_objects(entry, rel_lemma_to_objs, num_false_objects, model, tokenizer)

        # Generate a fact for each false object (and use the true one as parent fact)
        true_fact = {
            "subject": entry["subject"],
            "rel_lemma": entry["rel_lemma"],
            "object": entry["object"],
            "rel_p_id": entry["rel_p_id"],
            "query": entry["query"],
        }

        for false_object in entry["false_objects"]:
            false_fact = {
                "subject": entry["subject"],
                "rel_lemma": entry["rel_lemma"],
                "object": false_object,
                "rel_p_id": entry["rel_p_id"],
                "query": entry["query"],
                "fact_parent": true_fact,
            }

            extended_pararel.append(false_fact)

    # Show total number of false facts
    logger.info(f"Generated {len(extended_pararel)} false facts")

    return extended_pararel
