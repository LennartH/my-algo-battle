import random
import torch

from typing import List
from collections import namedtuple


MemoryEntry = namedtuple("MemoryEntry", field_names=("state", "action", "next_state", "reward"))


class Memory:

    def __init__(self, capacity: int):
        self._capacity = capacity
        self._index = 0
        self._entries: List[MemoryEntry] = []

    @property
    def capacity(self) -> int:
        return self._capacity

    def append(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor, reward: torch.Tensor):
        entry = MemoryEntry(state, action, next_state, reward)
        if len(self._entries) < self._capacity:
            self._entries.append(entry)
        else:
            self._entries[self._index] = entry
        self._index = (self._index + 1) % self._capacity

    def sample(self, size: int = None) -> List[MemoryEntry]:
        if size is None or size >= len(self._entries):
            return list(self._entries)
        else:
            return random.sample(self._entries, size)

    def __len__(self):
        return len(self._entries)
