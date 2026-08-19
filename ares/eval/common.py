from pathlib import Path
import torch
import numpy as np
import os
from ares.eval import pooling
from tqdm import tqdm
import torch.nn as nn
from ares.tokenization import AresProteinTokenizer
from ares.models import Ares


def freeze_backbone(model: nn.Module, enable: bool = True):
    if enable:
        for param in model.parameters():
            param.requires_grad = False


def identity_embedding_fn(x) -> torch.Tensor:
    return x


def ares_embedding_fn(x) -> torch.Tensor:
    return x.hidden_states[0]


def hf_embedding_fn(x) -> torch.Tensor:
    return x.last_hidden_state


def load_ares_model(checkpoint: str):
    tokenizer = AresProteinTokenizer()
    model = Ares.from_pretrained(checkpoint, device_map="cuda:0")
    return tokenizer, model


def load_esm_model(checkpoint: str):
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint, device_map="cuda:0")
    return tokenizer, model


def load_ankh_model(checkpoint: str):
    from transformers import AutoTokenizer, T5EncoderModel

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = T5EncoderModel.from_pretrained(checkpoint, device_map="cuda:0")
    return tokenizer, model


def load_model(model_name: str, checkpoint: str):
    if model_name == "ares":
        return load_ares_model(checkpoint)
    elif model_name == "esm":
        return load_esm_model(checkpoint)
    elif model_name == "ankh":
        return load_ankh_model(checkpoint)
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def get_extraction_function(model_name: str):
    if model_name == "ares":
        return ares_embedding_fn
    elif model_name in ["esm", "esm2", "ankh"]:
        return hf_embedding_fn
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def get_embed_dim(backbone):
    if hasattr(backbone.config, "embed_dim"):
        return backbone.config.embed_dim
    else:
        return backbone.config.hidden_size


def get_shifting_values(model_name: str):
    if model_name == "ankh":
        return 0, 1
    if model_name in ["ares", "esm", "esm2"]:
        return 1, 1
    raise ValueError(f"Invalid model name: {model_name}")


def extract_embeddings(model, tokenizer, dataset, extraction_function, pooler, output_path, shift_left, shift_right):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    pooler = pooling.get(pooler, embed_dim=get_embed_dim(model))
    model.eval()

    if len(list(pooler.parameters())) > 0:
        raise ValueError("Pooler must not have any learnable parameters")

    device = next(model.parameters()).device

    with torch.no_grad():
        for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc="Extracting embeddings"):
            path = os.path.join(output_path, "{}.npy".format(idx))
            if os.path.exists(path):
                continue
            sequence = example[dataset.sequences_column]
            input_ids = tokenizer.encode(sequence, add_special_tokens=True, return_tensors="pt")
            outputs = model(input_ids.to(device=device))

            embeddings = extraction_function(outputs)
            if shift_left is not None:
                embeddings = embeddings[:, shift_left:, :]
            if shift_right is not None:
                embeddings = embeddings[:, :-shift_right, :]
            # Poolers expect [batch, seq, dim] (single-sequence batch from encode).
            embeddings = pooler(embeddings).squeeze(0)
            np.save(path, embeddings.cpu().numpy())
