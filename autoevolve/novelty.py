from __future__ import annotations

import io
import tokenize

from .patching import evolve_content


_IGNORED_TOKENS = {
    tokenize.ENCODING,
    tokenize.ENDMARKER,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.NEWLINE,
    tokenize.NL,
    tokenize.COMMENT,
}


def _normalized_tokens(code: str) -> list[str]:
    mutable = evolve_content(code)
    try:
        stream = tokenize.generate_tokens(io.StringIO(mutable).readline)
        return [token.string for token in stream if token.type not in _IGNORED_TOKENS]
    except (IndentationError, tokenize.TokenError):
        return mutable.split()


def code_similarity(left: str, right: str, shingle_size: int = 5) -> float:
    """Token-shingle Jaccard similarity over mutable code only."""
    left_tokens = _normalized_tokens(left)
    right_tokens = _normalized_tokens(right)
    if left_tokens == right_tokens:
        return 1.0
    width = max(1, min(shingle_size, len(left_tokens), len(right_tokens)))
    left_shingles = {
        tuple(left_tokens[index : index + width])
        for index in range(max(1, len(left_tokens) - width + 1))
    }
    right_shingles = {
        tuple(right_tokens[index : index + width])
        for index in range(max(1, len(right_tokens) - width + 1))
    }
    union = left_shingles | right_shingles
    return 0.0 if not union else len(left_shingles & right_shingles) / len(union)
