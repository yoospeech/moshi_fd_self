"""Tokenizer helpers for full-duplex control symbols."""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase


CONTROL_TOKENS = ("<WAIT>", "<SPK>", "<HEAR>")


def add_control_tokens(tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
    """Register and return the IDs for the full-duplex control symbols.

    The operation is idempotent. ``add_special_tokens`` marks the symbols as
    special and returns stable IDs, while repeated calls leave the vocabulary
    unchanged.
    """
    tokenizer.add_special_tokens({"additional_special_tokens": list(CONTROL_TOKENS)})
    ids = {
        name[1:-1].lower(): int(tokenizer.convert_tokens_to_ids(name))
        for name in CONTROL_TOKENS
    }
    if len(set(ids.values())) != len(CONTROL_TOKENS):
        raise ValueError(f"control tokens did not receive distinct IDs: {ids}")
    for name, token_id in zip(CONTROL_TOKENS, ids.values(), strict=True):
        pieces = tokenizer.encode(name, add_special_tokens=False)
        if pieces != [token_id]:
            raise ValueError(f"{name} is not a single tokenizer token: {pieces}")
    return ids
