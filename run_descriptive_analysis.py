import argparse
from src.agent import HFAgent, OpenAIAgent
from src.descriptive_analysis.descriptive_analysis import generate_descriptive_analysis_answers
from src.utils.general import set_seed_everywhere
from src.utils.io import read_json

from src.utils.logger import freeze_args, get_logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fakepedia_path", type=str)
    parser.add_argument("--model_name_path", type=str)
    parser.add_argument("--temperature", default=0, type=int)
    parser.add_argument("--max_new_tokens", default=100, type=int)
    parser.add_argument("--top_p", default=1, type=int)
    parser.add_argument(
        "--system_message",
        type=str,
    )
    parser.add_argument(
        "--human_message_prompt_template",
        type=str,
    )
    parser.add_argument("--num_examples", type=int)
    parser.add_argument("--openai_api_key", default=None, type=str)
    parser.add_argument("--bfloat16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge_system_message", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume_path", default=None, type=str)
    parser.add_argument("--seed", default=100, type=int)

    return parser.parse_args()


def generate_answers(args):
    logger = get_logger()

    generation_parameters = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    logger.info("Initializing agent...")

    agent = (
        HFAgent(
            model_path=args.model_name_path,
            tokenizer_path=args.model_name_path,
            generation_parameters=generation_parameters,
            system_message=args.system_message,
            human_message_prompt_template=args.human_message_prompt_template,
            bfloat16=args.bfloat16,
            merge_system_message=args.merge_system_message,
            verbose=True,
        )
        if args.openai_api_key is None
        else OpenAIAgent(
            model_name=args.model_name_path,
            api_key=args.openai_api_key,
            generation_parameters=generation_parameters,
            system_message=args.system_message,
            human_message_prompt_template=args.human_message_prompt_template,
            verbose=True,
        )
    )

    logger.info("Loading base fakepedia...")

    base_fakepedia = read_json(args.fakepedia_path)

    logger.info("Generating descriptive analysis answers...")

    generate_descriptive_analysis_answers(base_fakepedia, agent, args.num_examples, resume_path=args.resume_path)


def main():
    args = get_args()
    freeze_args(args)
    set_seed_everywhere(args.seed)
    generate_answers(args)


if __name__ == "__main__":
    main()
