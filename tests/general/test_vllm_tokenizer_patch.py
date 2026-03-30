from swift.infer_engine.patch import _patch_tokenizer_compatibility


class DummyTokenizer:

    def __init__(self):
        self.all_special_tokens = ['<pad>', '<eos>']


def test_patch_tokenizer_compatibility_adds_all_special_tokens_extended():
    tokenizer = DummyTokenizer()

    _patch_tokenizer_compatibility(tokenizer)

    assert tokenizer.all_special_tokens_extended == tokenizer.all_special_tokens
