# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Hugging Face initialization for EasyMagpie's parallel text/phoneme channels."""

import torch
from omegaconf import OmegaConf, open_dict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def initialize_pretrained_hf_decoder(cfg, tokenizer, text_vocab_size, *, restoring=False):
    """Load an LLM and retain its text-token embedding rows.

    ``restoring`` constructs the saved architecture without fetching LLM weights:
    NeMo subsequently restores the trained decoder, embeddings and phoneme head.
    The tokenizer must have exactly the LLM's token-to-ID map at offset zero.
    EasyMagpie's own BOS/EOS/CFG/optional IPA rows are initialized separately,
    including when their IDs overlap unused, padded rows in the LLM checkpoint.
    """
    if cfg.get('decoder_type', 'huggingface') != 'huggingface':
        raise ValueError('transformer_hf_load_pretrained_weights requires decoder_type=huggingface')
    if cfg.get('disable_subword_embedding', False):
        raise ValueError('Pretrained HF initialization requires disable_subword_embedding=false')
    use_meta = cfg.get('use_meta_init_for_decoder', False)
    if use_meta and not restoring:
        raise ValueError('Meta initialization cannot replace pretrained weights outside NeMo restoration')

    source = cfg.transformer_hf_backend
    revision = cfg.get('transformer_hf_revision', None)
    saved_config = cfg.get('transformer_hf_config', None)
    if saved_config is None:
        config = AutoConfig.from_pretrained(source, revision=revision)
    else:
        config_dict = OmegaConf.to_container(saved_config, resolve=True)
        config = AutoConfig.for_model(config_dict.pop('model_type'), **config_dict)
    for name in ('embedding_dim', 'hidden_dim'):
        if cfg[name] != config.hidden_size:
            raise ValueError(f'{name}={cfg[name]} must match HF hidden_size={config.hidden_size}')
    if revision is None:
        revision = getattr(config, '_commit_hash', None)
    _validate_tokenizer(tokenizer, AutoTokenizer.from_pretrained(source, revision=revision, config=config))
    base_vocab_size = len(tokenizer.tokens)
    if config.vocab_size < base_vocab_size or text_vocab_size < base_vocab_size:
        raise ValueError('The pretrained embedding table must cover every base text-token ID')
    implementation = cfg.get('transformer_hf_attn_implementation', None)
    if implementation is not None:
        config._attn_implementation = implementation
    original_config = config.to_dict()

    if restoring:
        if use_meta:
            with torch.device('meta'):
                model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
            model = model.to_empty(device='cpu')
            model.tie_weights()
            for module in model.modules():
                if 'RotaryEmbedding' in type(module).__name__ and hasattr(module, 'config'):
                    type(module).__init__(module, module.config, device='cpu')
        else:
            model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    else:
        # Keep trainable master weights in FP32, like EasyMagpie's other modules.
        # Lightning bf16-mixed controls autocast; the checkpoint's BF16 storage
        # dtype must not silently quantize optimizer updates or NeMo restoration.
        model = AutoModelForCausalLM.from_pretrained(source, revision=revision, config=config, dtype=torch.float32)

    model.resize_token_embeddings(text_vocab_size, mean_resizing=False)
    with torch.no_grad():
        torch.nn.init.normal_(model.get_input_embeddings().weight[base_vocab_size:], std=config.initializer_range)
        output = model.get_output_embeddings()
        if output is not None and output.weight is not model.get_input_embeddings().weight:
            torch.nn.init.normal_(output.weight[base_vocab_size:], std=config.initializer_range)
    # Keep the ORIGINAL HF vocabulary size in the saved architecture. Restoring
    # repeats the same resize, then loads every trained EasyMagpie row from NeMo.
    with open_dict(cfg):
        if saved_config is None:
            cfg.transformer_hf_config = original_config
        cfg.transformer_hf_revision = revision
        if revision is not None:
            name = tokenizer.tokenizer_names[0]
            cfg.text_tokenizers[name].revision = revision
    return model, model.config


def _validate_tokenizer(tokenizer, reference):
    names = tokenizer.tokenizer_names
    if len(names) != 1 or tokenizer.tokenizer_offsets[names[0]] != 0:
        raise ValueError('Pretrained HF initialization requires one text tokenizer at vocabulary offset zero')
    actual_tokenizer = tokenizer.tokenizers[names[0]]
    if not hasattr(actual_tokenizer, 'get_vocab'):
        raise ValueError('Pretrained HF initialization requires its AutoTokenizer')
    actual = actual_tokenizer.get_vocab()
    if actual != reference.get_vocab() or set(actual.values()) != set(range(len(actual))):
        raise ValueError('Text tokenizer IDs must exactly match transformer_hf_backend; use its AutoTokenizer')
