"""UTF-8 byte tokenizer with exactly 256 IDs, including nanochat's chat markers."""

import codecs
import json
import os

from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS


class ByteTokenizer(RustBPETokenizer):
    """Reuse conversation rendering, with one token per UTF-8 byte and no merges.

    Bytes F5..FF never occur in valid UTF-8. Nine of these otherwise unused IDs
    represent the existing BOS/chat/tool markers, keeping the total vocabulary
    at 256 without colliding with any Unicode text (including control chars).
    """

    filename = "byte_tokenizer.json"

    def __init__(self):
        self.special_tokens = {name: 0xF5 + i for i, name in enumerate(SPECIAL_TOKENS)}
        self.bos_token_id = self.special_tokens["<|bos|>"]
        self._token_bytes = [bytes([i]) for i in range(256)]
        for name, token_id in self.special_tokens.items():
            self._token_bytes[token_id] = name.encode("utf-8")

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size=256):
        if vocab_size != 256:
            raise ValueError("The byte tokenizer requires exactly 256 vocabulary IDs")
        return cls() # no data-dependent training

    @classmethod
    def from_directory(cls, tokenizer_dir):
        with open(os.path.join(tokenizer_dir, cls.filename), encoding="utf-8") as f:
            metadata = json.load(f)
        tokenizer = cls()
        if metadata != tokenizer.metadata():
            raise ValueError(f"Unsupported byte tokenizer format: {metadata}")
        return tokenizer

    def metadata(self):
        return {"type": "utf8_byte", "version": 1, "vocab_size": 256,
                "special_tokens": self.special_tokens}

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        path = os.path.join(tokenizer_dir, self.filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metadata(), f, indent=2)
        print(f"Saved byte tokenizer to {path}")

    def get_vocab_size(self):
        return 256

    def get_special_tokens(self):
        return set(self.special_tokens)

    def encode_special(self, text):
        return self.special_tokens[text]

    def encode(self, text, prepend=None, append=None, num_threads=8):
        prepend_id = self.encode_special(prepend) if isinstance(prepend, str) else prepend
        append_id = self.encode_special(append) if isinstance(append, str) else append

        def encode_one(value):
            ids = list(value.encode("utf-8"))
            if prepend_id is not None:
                ids.insert(0, prepend_id)
            if append_id is not None:
                ids.append(append_id)
            return ids

        if isinstance(text, str):
            return encode_one(text)
        if isinstance(text, list):
            return [encode_one(value) for value in text]
        raise ValueError(f"Invalid input type: {type(text)}")

    def decode_single_token_bytes(self, token_id):
        return self._token_bytes[token_id]

    def decode(self, ids):
        return b"".join(self._token_bytes[i] for i in ids).decode("utf-8", errors="replace")

    def decoder(self):
        """Stateful UTF-8 decoder for streaming one byte at a time."""
        return codecs.getincrementaldecoder("utf-8")(errors="replace")

    def id_to_token(self, token_id):
        return self.decode([token_id])

    def token_byte_lengths(self, device="cpu"):
        import torch
        lengths = torch.ones(256, dtype=torch.int32, device=device)
        lengths[list(self.special_tokens.values())] = 0
        return lengths
