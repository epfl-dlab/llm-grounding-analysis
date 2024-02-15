import argparse

import torch
from src.causal_tracing.causal_tracing import run_causal_tracing_analysis
from src.utils.io import read_json

from src.utils.logger import freeze_args, get_logger

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fakepedia_path", type=str)
    parser.add_argument("--model_name_path", type=str)
    parser.add_argument("--prompt_template", default="Context: {context}\nAnswer: {query}", type=str)
    parser.add_argument("--num_grounded", type=int)
    parser.add_argument("--num_unfaithful", type=int)
    parser.add_argument("--prepend_space", action=argparse.BooleanOptionalAction)
    parser.add_argument("--bfloat16", action=argparse.BooleanOptionalAction)
    parser.add_argument("--resume_dir", default=None, type=str)

    return parser.parse_args()


def run_causal_tracing(args):
    logger = get_logger()

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_path, device_map="auto", torch_dtype=torch.bfloat16 if args.bfloat16 else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_path)

    logger.info("Loading fakepedia...")
    fakepedia = read_json(args.fakepedia_path)

    logger.info("Starting causal tracing...")
    run_causal_tracing_analysis(
        model,
        tokenizer,
        fakepedia,
        args.prompt_template,
        args.num_grounded,
        args.num_unfaithful,
        args.prepend_space,
        args.resume_dir,
    )


def main():
    args = get_args()
    freeze_args(args)
    run_causal_tracing(args)


if __name__ == "__main__":
    main()
