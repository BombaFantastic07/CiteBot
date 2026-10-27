"""Tiny dependency-free LRU cache for repeated questions."""
from __future__ import annotations

from collections import OrderedDict
from typing import Generic, Optional, TypeVar

V = TypeVar("V")


def normalize(question: str) -> str:
    return " ".join(question.lower().split())


class LRUCache(Generic[V]):
    def __init__(self, maxsize: int = 256) -> None:
        self.maxsize = maxsize
        self._data: "OrderedDict[str, V]" = OrderedDict()

    def get(self, key: str) -> Optional[V]:
        key = normalize(key)
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key: str, value: V) -> None:
        if self.maxsize <= 0:
            return
        key = normalize(key)
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)
