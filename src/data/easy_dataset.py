# Modified from https://github.com/CUT3R/CUT3R/blob/main/src/dust3r/datasets/base/easy_dataset.py

from typing import *

import numpy as np
from torch.utils.data import Dataset


class EasyDataset(Dataset):
    """A dataset that you can easily resize and combine.
    Examples:
    ---------
        2 * dataset ==> duplicate each element 2x

        10 @ dataset ==> set the size to 10 (random sampling, duplicates if necessary)

        dataset1 + dataset2 ==> concatenate datasets
    """

    def __add__(self, other: "EasyDataset"):
        return CatDataset([self, other])

    def __rmul__(self, factor: int):
        return MulDataset(factor, self)

    def __rmatmul__(self, factor: int):
        return ResizedDataset(factor, self)


class MulDataset(EasyDataset):
    """Artifically augmenting the size of a dataset."""

    multiplicator: int

    def __init__(self, multiplicator, dataset):
        assert isinstance(multiplicator, int) and multiplicator > 0
        self.multiplicator = multiplicator
        self.dataset = dataset

    def __len__(self):
        return self.multiplicator * len(self.dataset)

    def __getitem__(self, inputs: Union[int, Tuple[int, float, int]]):
        if not isinstance(inputs, tuple):
            index = inputs
            return self.dataset[index // self.multiplicator]
        else:
            index, aspect_ratio, image_num = inputs
            return self.dataset[index // self.multiplicator, aspect_ratio, image_num]


class ResizedDataset(EasyDataset):
    """Artifically changing the size of a dataset."""

    new_size: int

    def __init__(self, new_size: int, dataset: EasyDataset):
        assert isinstance(new_size, int) and new_size > 0
        self.new_size = new_size
        self.dataset = dataset

    def __len__(self):
        return self.new_size

    def __getitem__(self, inputs: Union[int, Tuple[int, float, int]]):
        if not isinstance(inputs, tuple):
            input_index = True
            index = inputs
        else:
            input_index = False
            index, aspect_ratio, image_num = inputs
        idxs = np.arange(len(self.dataset), dtype=np.int64)

        # Rotary extension until target size is met
        resized_idxs = np.concatenate(
            [idxs] * (1 + (len(self) - 1) // len(self.dataset))
        )
        self._idxs_mapping = resized_idxs[: self.new_size]
        assert len(self._idxs_mapping) == self.new_size

        if input_index:
            return self.dataset[self._idxs_mapping[index]]
        else:
            return self.dataset[self._idxs_mapping[index], aspect_ratio, image_num]


class CatDataset(EasyDataset):
    """Concatenation of several datasets"""

    def __init__(self, datasets: List[EasyDataset]):
        for dataset in datasets:
            assert isinstance(dataset, EasyDataset)
        self.datasets = datasets
        self._cum_sizes = np.cumsum([len(dataset) for dataset in datasets])

    def __len__(self):
        return self._cum_sizes[-1]

    def __getitem__(self, inputs: Union[int, Tuple[int, float, int]]):
        if not isinstance(inputs, tuple):
            input_index = True
            index = inputs
        else:
            input_index = False
            index, aspect_ratio, image_num = inputs

        assert 0 <= index < len(self)

        db_idx = np.searchsorted(self._cum_sizes, index, "right")
        dataset = self.datasets[db_idx]
        new_idx = index - (self._cum_sizes[db_idx - 1] if db_idx > 0 else 0)

        if input_index:
            return dataset[new_idx]
        else:
            return dataset[new_idx, aspect_ratio, image_num]
