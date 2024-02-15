import argparse
import os

from tqdm import tqdm
from src.fact import Fact, fact_from_dict
from src.utils.io import read_json, save_json

from src.utils.logger import freeze_args, get_logger
import re


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unfiltered_base_fakepedia_path", type=str
    )
    parser.add_argument("--data_dir", default="./data/", type=str)

    return parser.parse_args()


def split_text(text):
    # Regular expression pattern for commas and sentence enders
    pattern = r"[.?!]\s+|,\s+"

    parts = re.split(pattern, text)

    # Filter out any empty strings in the result
    parts = [part.strip() for part in parts if part.strip()]

    return parts


def is_paragraph_bad(fact: Fact):
    bad_expressions = [
        "often mistaken",
        "often misunderstood",
        "common misconception",
        "false",
        "is not",
        "was not",
        "does not",
        "did not",
    ]

    if fact.get_intermediate_paragraph() is not None:
        paragraph = fact.get_intermediate_paragraph().lower()
    else:
        paragraph = fact.get_paragraph().lower()

    false_object = fact.get_object().lower()
    true_object = fact.get_parent().get_object().lower()
    subject = fact.get_subject().lower()

    # Split paragraph into sentences on punctuation
    sentences = split_text(paragraph)

    # Check false object is in paragraph
    if false_object not in paragraph:
        return True

    # Check if any of the bad expressions are in the paragraph
    for sentence in sentences:
        # Check if sentence contains false object
        if false_object in sentence:
            # Check if sentence contains any of the expressions
            for expression in bad_expressions:
                if expression in sentence:
                    return True
        # Check if sentence contains true object, the subject and does not contain "not"
        # We do this only for facts where the true object is not part of the subject
        elif true_object not in subject and true_object in sentence and "not" not in sentence:
            return True

    return False


def generate_dataset(args):
    logger = get_logger()

    # Find good paragraphs
    bad_paragraphs = set()
    good_paragraphs = set()
    unfiltered_dataset = read_json(args.unfiltered_base_fakepedia_path)
    bad_paragraphs_count = 0
    for entry in tqdm(unfiltered_dataset, desc="Finding bad paragraphs"):
        fact = fact_from_dict(entry)

        to_discard = is_paragraph_bad(fact)

        if to_discard:
            bad_paragraphs.add(fact.get_paragraph())
            bad_paragraphs_count += 1
        else:
            good_paragraphs.add(fact.get_paragraph())

    # Show number of bad paragraphs and good paragraphs out of total
    logger.info(
        "Found {} bad paragraphs out of {} total paragraphs".format(bad_paragraphs_count, len(unfiltered_dataset))
    )

    # Get the files with "unfiltered_" in the absolute path and set the output path to the same path without "unfiltered_".
    files_to_filter = [
        os.path.join(dirpath, filename)
        for dirpath, dirnames, filenames in os.walk(args.data_dir)
        for filename in filenames
    ]
    files_to_filter = [file for file in files_to_filter if "unfiltered_" in file]

    for file_path in files_to_filter:

        output_path = file_path.replace("unfiltered_", "")

        logger.info("Loading '{}'...".format(file_path))

        unfiltered_dataset = read_json(file_path)

        logger.info("Filtering entries...")

        dataset = []
        for entry in tqdm(unfiltered_dataset, desc="Filtering entries"):
            fact = fact_from_dict(entry["fact"] if "fact" in entry else entry)

            if fact.get_intermediate_paragraph() is not None:
                paragraph = fact.get_intermediate_paragraph()
            else:
                paragraph = fact.get_paragraph()

            to_save = paragraph not in bad_paragraphs

            if to_save:
                dataset.append(entry)

        logger.info("Filtered dataset has {} entries".format(len(dataset)))
        logger.info("Saving filtered dataset to '{}'...".format(output_path))

        save_json(dataset, output_path)


def main():
    args = get_args()
    freeze_args(args)
    generate_dataset(args)


if __name__ == "__main__":
    main()
