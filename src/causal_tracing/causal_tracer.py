import torch
import torch.nn as nn
from src.model import find_submodule, get_embedding, get_next_token_probabilities
from tokenizers import Tokenizer

# Code partially adapted from https://github.com/kmeng01/rome

class MaskedCausalTracer:
    def __init__(self, model: nn.Module, tokenizer: Tokenizer, mask_token: str):
        self.device = next(model.parameters()).device
        self.model = model
        self.tokenizer = tokenizer
        self.mask_token = mask_token
        self.mask_token_embedding = self._get_mask_token_embedding(mask_token)

    def _get_mask_token_embedding(self, mask_token):
        token_attr = f"{mask_token}_token_id"
        if getattr(self.tokenizer, token_attr, None) is not None:
            mask_token_id = getattr(self.tokenizer, token_attr)
        else:
            raise ValueError("No such token in the tokenizer.")
        with torch.no_grad():
            corrupted_token_embedding = get_embedding(self.model, mask_token_id, self.device).clone()
        return corrupted_token_embedding

    def trace_with_patch(
        self,
        prompt,
        range_to_mask,  # A tuple (start, end) of tokens to corrupt
        target_tokens,  # Tokens whose probabilities we are interested in
        states_to_patch,  # A list of tuples (token index, modules) of states to restore
        embedding_module_name,  # Name of the embedding layer
    ):
        prompts = [prompt] * 2

        def untuple(x):
            return x[0] if isinstance(x, tuple) else x

        hooks = []

        # Add embedding hook
        def hook_embedding(module, input, output):
            output[1, range_to_mask[0] : range_to_mask[1]] = self.mask_token_embedding.clone()
            return output

        embedding_module = find_submodule(self.model, embedding_module_name)
        embedding_hook = embedding_module.register_forward_hook(hook_embedding)
        hooks.append(embedding_hook)

        # Add hooks for the modules to restore
        for token_to_restore, modules_to_restore in states_to_patch:
            for module_name in modules_to_restore:
                def restoring_hook(module, input, output):
                    h = untuple(output)
                    h[1, token_to_restore] = h[0, token_to_restore].clone()
                    return output

                module = find_submodule(self.model, module_name)
                module_hook = module.register_forward_hook(restoring_hook)
                hooks.append(module_hook)

        with torch.no_grad():
            probs = get_next_token_probabilities(self.model, self.tokenizer, prompts, target_tokens, device=self.device)
            clean_probs = probs[0, :]
            corrupted_probs = probs[1, :]

        for hook in hooks:
            hook.remove()

        return clean_probs, corrupted_probs
