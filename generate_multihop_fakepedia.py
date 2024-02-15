import argparse
import os
from src.fakepedia.data import generate_multihop_fakepedia
from src.utils.general import set_seed_everywhere
from src.utils.io import read_json, save_json

from src.utils.logger import freeze_args, get_logger, get_output_dir


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_fakepedia_path", default="./data/fakepedia/base_fakepedia.json", type=str)
    parser.add_argument("--extended_pararel_path", default="./data/pararel/extended_pararel.json", type=str)
    parser.add_argument("--linking_paragraph_dir", default="./data/pararel/raw/linking_templates", type=str)
    parser.add_argument("--max_matches", default=1, type=int)
    parser.add_argument("--seed", default=100, type=int)

    return parser.parse_args()


def generate_dataset(args):
    logger = get_logger()

    extended_pararel = read_json(args.extended_pararel_path)
    base_fakepedia = read_json(args.base_fakepedia_path)

    logger.info("Generating paragraphs...")
    multihop_fakepedia = generate_multihop_fakepedia(
        base_fakepedia, extended_pararel, args.max_matches, args.linking_paragraph_dir
    )

    logger.info("Saving paragraphs...")
    save_json(multihop_fakepedia, os.path.join(get_output_dir(), "multihop_fakepedia.json"))


def main():
    args = get_args()
    freeze_args(args)
    set_seed_everywhere(args.seed)
    generate_dataset(args)


if __name__ == "__main__":
    main()
