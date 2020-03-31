from functools import lru_cache
from pathlib import Path
from typing import Callable, Union

import torch
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer
from transformers.customs.common import get_specific_line, split_by_key, wc


@lru_cache(maxsize=None)
def get_line(path: Path, idx: int) -> str:
    return get_specific_line(path, idx)


class MisspelledDataset(Dataset):
    def __init__(self, datapath: Path, *transforms: Callable):
        assert datapath.exists(), f"Dataset does not exist in {datapath}"

        self.datapath = datapath
        self.transforms = transforms
        self._length = 0

    def __len__(self) -> int:
        return self.length

    @property
    def length(self) -> int:
        if not self._length:
            self._length = wc(self.datapath)

        return self._length

    def __getitem__(self, idx: Union[int, torch.Tensor]):
        if torch.is_tensor(idx):
            idx = idx.tolist()  # type: ignore

        output = get_line(self.datapath, idx)

        for transform in self.transforms:
            output = transform(output)

        return output


class DatasetTransformer:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
