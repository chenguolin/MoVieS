from typing import *
from torch.utils.data import DataLoader, Sampler

from .base_dataset import BaseDataset
from .dynamicreplica_dataset import DynamicreplicaDataset
from .matrixcity_dataset import MatrixcityDataset
from .pointodyssey_dataset import PointodysseyDataset
from .re10k_dataset import Re10kDataset
from .spring_dataset import SpringDataset
from .stereo4d_dataset import Stereo4dDataset
from .tartanair_dataset import TartanairDataset
from .vkitti2_dataset import Vkitti2Dataset


# Copied from https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/loader.py
class MultiEpochsDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.sampler) if self.batch_sampler is None else len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler: Sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def yield_forever(iterator: Iterator[Any]):
    while True:
        for x in iterator:
            yield x
