# Copyright (c) ModelScope Contributors. All rights reserved.
from contextlib import contextmanager
from functools import wraps
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase


def _patch_tokenizer_compatibility(tokenizer: PreTrainedTokenizerBase) -> None:
    if not hasattr(tokenizer, 'all_special_tokens_extended'):
        tokenizer.all_special_tokens_extended = tokenizer.all_special_tokens


@contextmanager
def patch_auto_tokenizer(tokenizer: PreTrainedTokenizerBase):
    _old_from_pretrained = AutoTokenizer.from_pretrained
    _patch_tokenizer_compatibility(tokenizer)

    @wraps(_old_from_pretrained)
    def _from_pretrained(*args, **kwargs):
        return tokenizer

    AutoTokenizer.from_pretrained = _from_pretrained
    try:
        yield
    finally:
        AutoTokenizer.from_pretrained = _old_from_pretrained


@contextmanager
def patch_auto_config(config: PretrainedConfig):
    _old_from_pretrained = AutoConfig.from_pretrained

    @wraps(_old_from_pretrained)
    def _from_pretrained(*args, **kwargs):
        return (config, {}) if 'return_unused_kwargs' in kwargs else config

    AutoConfig.from_pretrained = _from_pretrained
    try:
        yield
    finally:
        AutoConfig.from_pretrained = _old_from_pretrained
