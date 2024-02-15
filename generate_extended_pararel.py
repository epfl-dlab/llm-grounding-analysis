import argparse
import os
from src.pararel.data import extract_data, process_raw_data
from src.utils.general import set_seed_everywhere
from src.utils.io import save_json

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.logger import freeze_args, get_logger, get_output_dir


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pararel_dir", type=str)
    parser.add_argument("--model_name_path", type=str)
    parser.add_argument("--num_false_objects", default=4, type=int)
    parser.add_argument("--seed", default=100, type=int)

    return parser.parse_args()


def generate_extended_pararel(args):
    logger = get_logger()

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_path)

    logger.info("Extracting data from ParaRel...")
    raw_data = extract_data(args.pararel_dir)

    logger.info("Processing raw data...")
    extended_pararel = process_raw_data(raw_data, model, tokenizer, args.num_false_objects)

    logger.info("Saving extended pararel...")
    output_dir = get_output_dir()
    extended_pararel_path = os.path.join(output_dir, "extended_pararel.json")
    save_json(extended_pararel, extended_pararel_path)


def main():
    args = get_args()
    freeze_args(args)
    set_seed_everywhere(args.seed)
    generate_extended_pararel(args)


if __name__ == "__main__":
    main()
