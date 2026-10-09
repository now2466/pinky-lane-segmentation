"""Repeat rare supervised cases without omitting any original training image."""
import torch
from torch.utils.data import Sampler


class CompleteRepeatSampler(Sampler):
    def __init__(self, repeats, seed):
        if not repeats or any(not isinstance(n, int) or not 1 <= n <= 10 for n in repeats):
            raise ValueError('Repeat counts must be integers in [1,10]')
        self.indices = [i for i, n in enumerate(repeats) for _ in range(n)]
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        order = torch.randperm(len(self.indices), generator=self.generator).tolist()
        return iter(self.indices[i] for i in order)
