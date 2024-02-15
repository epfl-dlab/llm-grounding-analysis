import re
from typing import Union, List
import torch
from operator import attrgetter

# Code partially adapted from https://github.com/kmeng01/rome

def get_next_token(model, tokenizer, prompt, device):
    # Prepare inputs
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)

    # Feed model
    with torch.no_grad():
        next_token_logits = model(input_ids=prompt_ids)["logits"].detach()[0, -1, :]

    # Find the token with the highest probability and its logit
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    max_prob, max_prob_index = torch.max(next_token_probs, dim=-1)
    max_prob = max_prob.item()
    max_prob_index = max_prob_index.item()

    # Convert the token index back to a string
    token = tokenizer.convert_ids_to_tokens(max_prob_index)

    return token, max_prob


def get_next_token_probabilities(
    model, tokenizer, prompts: Union[str, List[str]], target_tokens: Union[str, List[str]], device
):
    # Make input as list of strings if a single string was given
    if type(prompts) == str:
        prompts = [prompts]

    if type(target_tokens) == str:
        target_tokens = [target_tokens]

    # Prepare inputs
    encoded_prompts = tokenizer(
        prompts,
        return_tensors="pt",
        return_token_type_ids=False,
    ).to(device)
    target_token_ids = [tokenizer.convert_tokens_to_ids(next_token) for next_token in target_tokens]

    # Feed model
    with torch.no_grad():
        next_token_logits = model(**encoded_prompts)["logits"][:, -1, :]

    # Extract target token logits and probabilities
    target_token_probs = torch.softmax(next_token_logits, dim=-1)[:, target_token_ids]

    return target_token_probs


def adapt_target_tokens(tokenizer, target_tokens: List[str], preprend_space: bool):
    """
    Make sure that target_tokens contain correspond to only a single token
    """
    if preprend_space:
        target_tokens = [" " + token.lstrip() for token in target_tokens]

    target_tokens = [tokenizer.tokenize(token)[0] for token in target_tokens]

    return target_tokens


def find_substring_range(tokenizer, string, substring):
    string_ids = tokenizer(
        string,
        return_tensors=None,
        return_token_type_ids=False,
    )["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(string_ids)
    string = "".join(tokens)

    substring_ids = tokenizer.tokenize(substring)
    substring = "".join(substring_ids)

    char_loc = string.rindex(substring)
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(tokens):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break

    return tok_start, tok_end


def get_module_name(model, kind, num=None):
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind == "hidden" else "." + kind}'
    if hasattr(model, "model"):
        if kind == "embed":
            return "model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f'model.layers.{num}{"" if kind == "hidden" else "." + kind}'
    assert False, "unknown transformer structure"


def get_num_layers(model):
    return len([n for n, m in model.named_modules() if (re.match(r"^(transformer|model)\.(h|layers)\.\d+$", n))])


def get_num_tokens(tokenizer, string):
    tokens_ids = tokenizer(
        string,
        return_tensors=None,
        return_token_type_ids=False,
    )["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(tokens_ids)
    # print("tokens: \n", list(zip(range(len(tokens)), tokens)))
    return len(tokens)


def find_submodule(module, name):
    """
    Finds the named module within the given model.
    """
    for n, m in module.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def get_embedding(model, token_id, device):
    # Prepare inputs
    token_ids = torch.tensor([[token_id]], device=device)

    # Feed model
    embed_module = attrgetter(get_module_name(model, "embed", 0))(model)
    embedding = embed_module(token_ids)[0, 0, :]

    return embedding
