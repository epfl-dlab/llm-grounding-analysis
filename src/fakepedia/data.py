import copy
import json
import os
from random import sample
from typing import Dict, List

from tqdm import tqdm

from src.agent import Agent
from src.fact import fact_from_dict
from src.utils.context import ResumeAndSaveFactDataset
from src.utils.logger import get_logger, get_output_dir


def generate_base_fakepedia(extended_pararel: List[Dict], agent: Agent, num_examples: int, resume_path: str = None):
    if resume_path is None:
        resume_path = os.path.join(get_output_dir(), "base_fakepedia.json")

    with ResumeAndSaveFactDataset(resume_path) as output_dataset:
        for entry in tqdm(extended_pararel[:num_examples], desc="Generating fakepedia paragraphs"):

            fact = fact_from_dict(entry)

            if output_dataset.is_input_processed(fact):
                continue

            response = agent({"query": f"{fact.get_query()} {fact.get_object()}"})
            paragraph = response["response_content"]

            new_entry = copy.deepcopy(entry)
            new_entry["fact_paragraph"] = paragraph

            output_dataset.add_entry(new_entry)


def generate_multihop_fakepedia(
    fakepedia: List[Dict], extended_pararel: List[Dict], max_matches: int, linking_paragraphs_dir: str
):
    # Open and index linking paragraphs by rel_p_id and rel_lemma
    # 1. List files in the linking_paragraphs_dir. Each file is named as rel_p_id.json
    # 2. For each file, load the linking paragraphs and index them by rel_lemma
    # 3. Index the linking paragraphs by rel_p_id

    linking_paragraphs_files = os.listdir(linking_paragraphs_dir)
    acceptable_rel_p_ids = [file.split(".")[0] for file in linking_paragraphs_files]
    linking_paragraphs = {}
    for rel_p_id, linking_paragraph_file in zip(acceptable_rel_p_ids, linking_paragraphs_files):
        # All files are jsonl files
        # {"pattern": "[X] died in the same moment as [Y].", "lemma": "die", "extended_lemma": "die-in", "tense": "past"}
        # Some lemmas are repeated (meaning that we should create lists of paragraphs for each lemma)

        with open(os.path.join(linking_paragraphs_dir, linking_paragraph_file)) as f:
            linking_paragraphs[rel_p_id] = {}
            for line in f:
                linking_paragraph = json.loads(line.strip())
                rel_lemma = linking_paragraph["extended_lemma"]
                if rel_lemma not in linking_paragraphs[rel_p_id]:
                    linking_paragraphs[rel_p_id][rel_lemma] = []
                linking_paragraphs[rel_p_id][rel_lemma].append(linking_paragraph["pattern"])

    # Assuming linking_paragraphs is a dictionary of dictionaries structured as {rel_p_id: {rel_lemma: paragraph}}
    total_potential_matches = 0
    total_facts = 0
    facts_without_matches = 0

    multihop_fakepedia = []

    # Iterate over each base Fakepedia fact
    for base_fakepedia_fact in tqdm(fakepedia, desc="Generating multihop fakepedia paragraphs"):
        total_facts += 1

        base_fakepedia_fact_object = base_fakepedia_fact["object"]
        base_fakepedia_fact_rel_lemma = base_fakepedia_fact["rel_lemma"]
        base_fakepedia_fact_p_rel_id = base_fakepedia_fact["rel_p_id"]
        base_fakepedia_fact_subject = base_fakepedia_fact["subject"]

        # Filtering extended Pararel facts
        matching_facts = [
            fact
            for fact in extended_pararel
            if fact["object"] == base_fakepedia_fact_object
            and fact["rel_lemma"] == base_fakepedia_fact_rel_lemma
            and fact["rel_p_id"] == base_fakepedia_fact_p_rel_id
            and fact["subject"] != base_fakepedia_fact_subject
            and fact["rel_p_id"] in acceptable_rel_p_ids
        ]

        total_potential_matches += len(matching_facts)

        if len(matching_facts) == 0:
            facts_without_matches += 1
            continue

        # Randomly selecting at most max_matches from the filtered facts
        selected_facts = sample(matching_facts, min(len(matching_facts), max_matches))

        # Creating multihop paragraphs
        for fact in selected_facts:
            linking_paragraph_template = linking_paragraphs[fact["rel_p_id"]][fact["rel_lemma"]][0]

            # Replace [X] with the subject of the fact and [Y] with the subject of the base fakepedia fact
            linking_paragraph = linking_paragraph_template.replace("[X]", fact["subject"]).replace(
                "[Y]", base_fakepedia_fact_subject
            )

            # Appending only the formatted linking paragraph to create the extended paragraph
            extended_paragraph = base_fakepedia_fact["fact_paragraph"] + "\n" + linking_paragraph

            # Make a copy of the fact and add the extended paragraph, then append it to the multihop fakepedia
            new_fact = copy.deepcopy(fact)
            new_fact["fact_paragraph"] = extended_paragraph
            new_fact["intermediate_paragraph"] = base_fakepedia_fact["fact_paragraph"]
            multihop_fakepedia.append(fact_from_dict(new_fact).as_dict())

    logger = get_logger()
    logger.info(f"Total facts: {total_facts}")
    logger.info(f"Total potential matches: {total_potential_matches}")
    logger.info(f"Facts without matches: {facts_without_matches}")
    logger.info(f"Multihop fakepedia size: {len(multihop_fakepedia)}")

    return multihop_fakepedia
