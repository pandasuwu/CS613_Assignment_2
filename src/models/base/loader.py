from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from src.config import get_device, logger


def load_base_model(model_id: str) -> tuple[Any, Any]:
    """Load an un-fine-tuned base causal language model and tokenizer onto target accelerator."""
    device = get_device()
    logger.info("Loading base LLM '%s' onto device '%s'", model_id, device)

    tokenizer: Any = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    model: Any = AutoModel.from_pretrained(
        model_id,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    return model, tokenizer
