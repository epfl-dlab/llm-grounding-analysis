import os
import numpy as np
from tokenizers import Tokenizer
import torch
from tqdm import tqdm
from src.causal_tracing.causal_tracer import MaskedCausalTracer
from src.fact import Fact, fact_from_dict

from src.model import (
    adapt_target_tokens,
    find_substring_range,
    get_module_name,
    get_next_token,
    get_num_layers,
    get_num_tokens,
)

from torch import nn

from src.utils.context import ResumeAndSaveFactDataset
from src.utils.io import read_json
from src.utils.logger import get_logger, get_output_dir

# Code partially adapted from https://github.com/kmeng01/rome

class Feature:
    def __init__(self, name):
        self.name = name
        self.d = []

    def get_name(self):
        return self.name

    def to_array(self):
        return np.array(self.d)

    def add(self, v):
        self.d.append(v)

    def avg(self):
        np_array = np.array(self.d)
        return np.mean(np_array[~np.isnan(np_array)])

    def std(self):
        np_array = np.array(self.d)
        return np.std(np_array[~np.isnan(np_array)])

    def __len__(self):
        return len(self.d)

    def get(self, i):
        return self.d[i]


def group_results(facts, bucket):
    labels = ["subj-first", "subj-middle", "subj-last", "cont-first", "cont-middle", "cont-last"]

    corrupted_probs = Feature("corr")
    clean_probs = Feature("clean")
    results = {kind: {labels[i]: Feature(labels[i]) for i in range(6)} for kind in ["hidden", "mlp", "attn"]}

    target_token = f"{bucket}_token"

    for processed_fact in facts:

        processed_fact = processed_fact["results"]
        corrupted_score = processed_fact["corrupted"][target_token]["probs"]
        clean_score = processed_fact["clean"][target_token]["probs"]

        # If there is a zero interval, skip the fact
        interval_to_explain = max(clean_score - corrupted_score, 0)
        if interval_to_explain == 0:
            continue

        corrupted_probs.add(corrupted_score)
        clean_probs.add(clean_score)

        for kind in ["hidden", "mlp", "attn"]:
            (
                avg_first_subject,
                avg_middle_subject,
                avg_last_subject,
                avg_first_after,
                avg_middle_after,
                avg_last_after,
            ) = results[kind].values()

            tokens = processed_fact["tokens"]
            started_subject = False
            finished_subject = False
            temp_mid = 0.0
            count_mid = 0

            for token in tokens:
                interval_explained = max(token[kind][target_token]["probs"] - corrupted_score, 0)
                token_effect = min(interval_explained / interval_to_explain, 1)

                if "subject_pos" in token:
                    if not started_subject:
                        avg_first_subject.add(token_effect)
                        started_subject = True

                        if token["subject_pos"] == -1:
                            avg_last_subject.add(token_effect)
                    else:
                        subject_pos = token["subject_pos"]
                        if subject_pos == -1:
                            avg_last_subject.add(token_effect)
                        else:
                            temp_mid += token_effect
                            count_mid += 1
                else:
                    if not finished_subject:
                        # Process all subject middle tokens
                        if count_mid > 0:
                            avg_middle_subject.add(temp_mid / count_mid)
                            temp_mid = 0.0
                            count_mid = 0
                        else:
                            avg_middle_subject.add(0.0)
                        avg_first_after.add(token_effect)
                        finished_subject = True

                        if token["pos"] == -1:
                            avg_last_after.add(token_effect)
                    else:
                        token_pos = token["pos"]
                        if token_pos == -1:
                            avg_last_after.add(token_effect)
                        else:
                            temp_mid += token_effect
                            count_mid += 1

            if count_mid > 0:
                avg_middle_after.add(temp_mid / count_mid)
            else:
                avg_middle_after.add(0.0)

    return results, corrupted_probs, clean_probs


def process_entry(causal_tracer: MaskedCausalTracer, prompt: str, subject: str, target_token: str, bucket: str):

    output = dict()

    embedding_module_name = get_module_name(causal_tracer.model, "embed", 0)
    subject_tokens_range = find_substring_range(causal_tracer.tokenizer, prompt, subject)

    # Get corrupted run results
    clean_probs, corrupted_probs = causal_tracer.trace_with_patch(
        prompt, subject_tokens_range, [target_token], [(None, [])], embedding_module_name
    )
    corrupted_output = {"token": target_token, "probs": corrupted_probs[0].item()}
    clean_output = {"token": target_token, "probs": clean_probs[0].item()}

    output["results"] = {
        "corrupted": {
            f"{bucket}_token": corrupted_output,
        },
        "clean": {
            f"{bucket}_token": clean_output,
        },
    }

    # Get patched runs results
    num_tokens = get_num_tokens(causal_tracer.tokenizer, prompt)
    output["results"]["tokens"] = list()

    # We start the loop from the first subject token as patching previous tokens has no effect
    for token_i in range(subject_tokens_range[0], num_tokens):
        output["results"]["tokens"].append({"pos": token_i - num_tokens})

        # If token is part of the subject, store its relative negative position
        if subject_tokens_range[0] <= token_i < subject_tokens_range[1]:
            output["results"]["tokens"][-1]["subject_pos"] = token_i - subject_tokens_range[1]

        for kind in ["hidden", "mlp", "attn"]:
            states_to_patch = (
                token_i,
                [
                    get_module_name(causal_tracer.model, kind, L)
                    for L in range(
                        0,
                        get_num_layers(causal_tracer.model),
                    )
                ],
            )
            _, patched_probs = causal_tracer.trace_with_patch(
                prompt, subject_tokens_range, [target_token], [states_to_patch], embedding_module_name
            )
            patched_output = {"token": target_token, "probs": patched_probs[0].item()}
            patched_results = {
                f"{bucket}_token": patched_output,
            }
            output["results"]["tokens"][-1][kind] = patched_results

    return output


def construct_prompt(fact: Fact, prompt_template):
    prompt = prompt_template.format(query=fact.get_query(), context=fact.get_paragraph())
    return prompt


def run_causal_tracing_analysis(
    model: nn.Module,
    tokenizer: Tokenizer,
    fakepedia,
    prompt_template,
    num_grounded,
    num_unfaithful,
    prepend_space,
    resume_dir,
):
    # We keep the results in two different files: unfaithful and grounded
    #
    # For each fact:
    #
    # Verify if the answer of the model is the unfaithful object or the grounded object. If the answer is another token, then skip the fact.
    # Put the fact in the corresponding list.
    #
    # Once we have processed all the facts, for each list and for each fact of the list we run the causal tracer.
    # Finally, we save the results in the corresponding file.

    device = next(model.parameters()).device
    logger = get_logger()

    if resume_dir is None:
        resume_dir = get_output_dir()
    os.makedirs(resume_dir, exist_ok=True)

    partial_path = os.path.join(resume_dir, "partial.json")

    with ResumeAndSaveFactDataset(partial_path) as partial_dataset:
        for entry in tqdm(fakepedia, desc="Filtering facts"):
            fact = fact_from_dict(entry)
            if partial_dataset.is_input_processed(fact):
                continue

            # Adapt unfaithful and grounded objects
            target_tokens = adapt_target_tokens(
                tokenizer, [fact.get_parent().get_object(), fact.get_object()], prepend_space
            )

            # Predict most likely next token
            prompt = construct_prompt(fact, prompt_template)
            most_likely_next_token, _ = get_next_token(model, tokenizer, prompt, device=device)

            partial_dataset.add_entry(
                {
                    "fact": fact.as_dict(),
                    "partial_results": {
                        "prompt": prompt,
                        "next_token": most_likely_next_token,
                        "is_unfaithful": most_likely_next_token == target_tokens[0],
                        "is_grounded": most_likely_next_token == target_tokens[1],
                    },
                }
            )

    partial_dataset = read_json(partial_path)

    unfaithful_facts = []
    grounded_facts = []

    for entry in partial_dataset:
        if entry["partial_results"]["is_grounded"]:
            grounded_facts.append(entry)
        elif entry["partial_results"]["is_unfaithful"]:
            unfaithful_facts.append(entry)

    logger.info(f"Found {len(unfaithful_facts)} unfaithful facts and {len(grounded_facts)} grounded facts")

    causal_tracer = MaskedCausalTracer(model, tokenizer, "eos")

    for bucket in ["grounded", "unfaithful"]:
        if bucket == "unfaithful":
            if num_unfaithful == -1:
                num_unfaithful = len(unfaithful_facts)
            facts = unfaithful_facts[:num_unfaithful]
        else:
            if num_grounded == -1:
                num_grounded = len(grounded_facts)
            facts = grounded_facts[:num_grounded]

        num_facts = len(facts)

        causal_traces_path = os.path.join(resume_dir, f"{bucket}.json")

        logger.info(f"Running causal tracing on {num_facts} {bucket} facts")
        with ResumeAndSaveFactDataset(causal_traces_path, save_interval=1000) as dataset:
            for entry in tqdm(facts, desc=f"Running causal tracing on {bucket} facts"):

                fact = fact_from_dict(entry["fact"])

                if dataset.is_input_processed(fact):
                    continue

                prompt = entry["partial_results"]["prompt"]
                target_token = entry["partial_results"]["next_token"]

                output_entry = process_entry(causal_tracer, prompt, fact.get_subject(), target_token, bucket)

                output_entry["fact"] = fact.as_dict()

                dataset.add_entry(output_entry)
