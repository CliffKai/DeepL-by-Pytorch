# src/bpe/tokenizer.py
from __future__ import annotations
from dataclasses import dataclass
import regex as re
from typing import Dict, Iterable, Iterator, List, Tuple, Optional

def get_tokenizer(
    vocab: Dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
    special_tokens: Optional[List[str]] = None,
):
    return _Tokenizer(vocab, merges, special_tokens or [])


@dataclass
class _Tokenizer:
    PAT_STR = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], specials: List[str]) -> None:
        self.pat = re.compile(self.PAT_STR)

        self.id_to_bytes = dict(vocab)
        self._bytes_to_id: Dict[bytes, int] = {v: k for k, v in self.id_to_bytes.items()}
        self._rank: Dict[Tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}
        
        self.special_tokens = specials
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "|".join(map(re.escape, sorted_specials))
            self.special_pat = re.compile(f"({special_pattern})")
        else:
            self.special_pat = None

    def encode(self, text: str) -> List[int]:
        return list(self._encode_generator(text))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self._encode_generator(chunk)

    def decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        b = b"".join(self.id_to_bytes[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    def _encode_generator(self, text: str) -> Iterator[int]:
        if not text:
            return
        
        if self.special_pat is None:
            chunks = [text]
        else:
            chunks = self.special_pat.split(text)

        for chunk in chunks:
            if not chunk:
                continue
            
            if self.special_pat and self.special_pat.fullmatch(chunk):
                yield self._bytes_to_id[chunk.encode("utf-8")]
            else:
                pre_tokens = self.pat.findall(chunk)
                for pre_token in pre_tokens:
                    pre_token_bytes = pre_token.encode("utf-8")
                    yield from self._bpe_segment(pre_token_bytes)

    def _bpe_segment(self, b: bytes) -> Iterator[int]:
        if not b:
            return

        tokens: List[bytes] = [bytes([byte]) for byte in b]

        if len(tokens) == 1:
            yield self._bytes_to_id[tokens[0]]
            return

        while len(tokens) > 1:
            pairs = zip(tokens, tokens[1:])
            best_pair = min(
                (
                    (self._rank.get(pair, float('inf')), i)
                    for i, pair in enumerate(pairs)
                ),
                key=lambda x: x[0]
            )

            best_rank, merge_idx = best_pair

            if best_rank == float('inf'):
                break

            tokens[merge_idx : merge_idx+2] = [tokens[merge_idx] + tokens[merge_idx+1]]
        
        for token in tokens:
            yield self._bytes_to_id[token]