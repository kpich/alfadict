"""8-gram near-duplicate cache backed by a numpy .npy file.

The cache stores SHA256[:8]-derived int64 hashes of word 8-grams sampled
from each document (every 10th gram when indexing, all grams when querying).
"""

import hashlib
from pathlib import Path

import numpy as np


class NgramCache:
    def __init__(self) -> None:
        self._hashes: set[int] = set()

    @classmethod
    def load(cls, path: Path) -> "NgramCache":
        cache = cls()
        arr = np.load(path)
        cache._hashes = set(arr.tolist())
        return cache

    def save(self, path: Path) -> None:
        arr = np.array(list(self._hashes), dtype=np.int64)
        np.save(path, arr)

    def _hash_gram(self, gram: str) -> int:
        h = hashlib.sha256(gram.encode()).digest()[:8]
        return int.from_bytes(h, "little", signed=True)

    def _word_8grams(self, text: str) -> list[str]:
        words = text.split()
        return [" ".join(words[i : i + 8]) for i in range(len(words) - 7)]

    def add_doc(self, text: str) -> None:
        """Index a document: sample every 10th 8-gram."""
        grams = self._word_8grams(text)
        for i, gram in enumerate(grams):
            if i % 10 == 0:
                self._hashes.add(self._hash_gram(gram))

    def add_docs(self, texts: list[str]) -> None:
        for text in texts:
            self.add_doc(text)

    def is_near_duplicate(self, text: str, threshold: float = 0.05) -> bool:
        """Return True if >= threshold fraction of 8-grams match the cache."""
        grams = self._word_8grams(text)
        if not grams:
            return False
        hits = sum(1 for gram in grams if self._hash_gram(gram) in self._hashes)
        return hits / len(grams) >= threshold
