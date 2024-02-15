import os
from typing import Dict, List

from tqdm import tqdm

from src.agent import Agent
from src.fact import Fact, fact_from_dict
from src.utils.context import ResumeAndSaveFactDataset
from src.utils.logger import get_output_dir


def generate_descriptive_analysis_answers(
    fakepedia: List[Dict], agent: Agent, num_examples: int, resume_path: str = None
):
    if resume_path is None:
        resume_path = os.path.join(get_output_dir(), f"{agent.get_name()}.json")

    # Create the output directory if it does not exist
    os.makedirs(os.path.dirname(resume_path), exist_ok=True)

    with ResumeAndSaveFactDataset(resume_path) as output_dataset:
        for entry in tqdm(fakepedia[:num_examples], desc="Generating descriptive analysis answers"):

            fact = fact_from_dict(entry)

            if output_dataset.is_input_processed(fact):
                continue

            responses = get_responses(agent, fact)

            new_entry = {
                "fact": entry,
                "answers": responses,
            }

            output_dataset.add_entry(new_entry)


def get_responses(agent: Agent, fact: Fact):
    responses = []

    query = fact.get_query()
    paragraph = fact.get_paragraph()

    grounded_object = fact.get_object()
    unfaithful_object = fact.get_parent().get_object()

    # Step 3.a: feed the agent with the grounded object as option A
    option_a_grounded_entry = get_agent_output(
        agent, query, paragraph, option_a=grounded_object, option_b=unfaithful_object
    )
    if option_a_grounded_entry["answer"] is None:
        option_a_grounded_entry["grounded"] = None
    else:
        option_a_grounded_entry["grounded"] = option_a_grounded_entry["answer"] == "A"

    # Step 3.b: feed the agent with the grounded object as option B
    option_b_grounded_entry = get_agent_output(
        agent, query, paragraph, option_a=unfaithful_object, option_b=grounded_object
    )
    if option_b_grounded_entry["answer"] is None:
        option_b_grounded_entry["grounded"] = None
    else:
        option_b_grounded_entry["grounded"] = option_b_grounded_entry["answer"] == "B"

    # Step 4: save the results in the QA dataset
    responses = {
        "option_a_grounded": option_a_grounded_entry,
        "option_b_grounded": option_b_grounded_entry,
    }
    return responses


def get_agent_output(agent: Agent, query: str, paragraph: str, option_a: str, option_b: str):
    response = agent(
        {
            "query": query,
            "context": paragraph,
            "option_a": option_a,
            "option_b": option_b,
        }
    )

    answer = response["response_content"].strip()

    if "The correct answer is A".upper() in answer.upper():
        answer = "A"
    elif "The correct answer is B".upper() in answer.upper():
        answer = "B"
    else:
        answer = None

    return {"answer": answer, "long_answer": response["response_content"]}
