import argparse
from src.agent import OpenAIAgent
from src.fakepedia.data import generate_base_fakepedia
from src.utils.general import set_seed_everywhere
from src.utils.io import read_json

from src.utils.logger import freeze_args, get_logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended_pararel_path", type=str)
    parser.add_argument("--model_name_path", type=str)
    parser.add_argument("--temperature", type=int)
    parser.add_argument("--max_new_tokens", type=int)
    parser.add_argument("--top_p", type=int)
    parser.add_argument(
        "--system_message", type=str
    )
    parser.add_argument("--human_message_prompt_template", type=str)
    parser.add_argument("--num_examples", type=int)
    parser.add_argument("--openai_api_key", default=None, type=str)
    parser.add_argument("--resume_path", default=None, type=str)
    parser.add_argument("--seed", default=100, type=int)

    return parser.parse_args()


def generate_dataset(args):
    logger = get_logger()

    generation_parameters = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    logger.info("Initializing agent...")

    agent = OpenAIAgent(
        model_name=args.model_name_path,
        api_key=args.openai_api_key,
        generation_parameters=generation_parameters,
        system_message=args.system_message,
        human_message_prompt_template=args.human_message_prompt_template,
        verbose=True,
    )

    extended_pararel = read_json(args.extended_pararel_path)

    logger.info("Generating paragraphs...")

    generate_base_fakepedia(extended_pararel, agent, args.num_examples, resume_path=args.resume_path)


def main():
    args = get_args()
    freeze_args(args)
    set_seed_everywhere(args.seed)
    generate_dataset(args)


if __name__ == "__main__":
    main()
