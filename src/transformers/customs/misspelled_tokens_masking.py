import itertools
import random
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from misspell import make_typo
from torch.utils.data import DataLoader, Dataset

from transformers.customs.common import get_specific_line, split_by_key, wc
from transformers.tokenization_utils import PreTrainedTokenizer


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
            idx = idx.tolist()

        output = get_line(self.datapath, idx)

        for transform in self.transforms:
            output = transform(output)

        return output


class BaseMisspellTransform:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer


class SentenceSplitTransform(BaseMisspellTransform):
    def __call__(self, text: str) -> Dict[str, np.ndarray]:
        words = np.array(text.split(), dtype=np.str)
        word_tokens = [self.tokenizer.tokenize(word) for word in words]
        # word_tokens_ids = np.array(
        #     list(
        #         itertools.chain.from_iterable(
        #             [idx] * len(tokens) for idx, tokens in enumerate(word_tokens)
        #         )
        #     ),
        #     dtype=np.int,
        # )
        tokens = np.array(list(itertools.chain.from_iterable(word_tokens)), dtype=np.str)

        return dict(tokens=tokens)


class SentenceMaskingTransform(BaseMisspellTransform):
    def __init__(self, tokenizer: PreTrainedTokenizer, mask_tokens_ratio: float = 0.15):
        super(SentenceMaskingTransform, self).__init__(tokenizer)
        self.mask_tokens_ratio = mask_tokens_ratio

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        no_tokens = len(sample["tokens"])

        no_tokens_to_draw = int(self.mask_tokens_ratio * no_tokens)
        mask_ids = random.sample(list(range(1, no_tokens - 1)), no_tokens_to_draw)
        mask_ids = self._make_mask_not_consecutive(mask_ids)

        mask = np.full(no_tokens, False)
        mask[mask_ids] = True

        masked_tokens = sample["tokens"].copy()
        masked_tokens[mask] = self.tokenizer.mask_token

        return dict(**sample, mask=mask, masked_tokens=masked_tokens)

    def _make_mask_not_consecutive(self, mask_ids: np.ndarray) -> np.ndarray:
        return mask_ids


class MisspellTransform(BaseMisspellTransform):
    def __init__(self, tokenizer: PreTrainedTokenizer, no_misspell: int = 3):
        super(MisspellTransform, self).__init__(tokenizer)
        self.no_misspell = no_misspell

    def __call__(self, sample: Dict[str, np.ndarray]):
        splits = split_by_key(sample["masked_tokens"], self.tokenizer.mask_token, skip_key=False)

        decoded_parts = [self.tokenizer.convert_tokens_to_string(split) for split in splits]
        decoded_parts = [self.tokenizer.mask_token if not text else text for text in decoded_parts]
        texts_ids = np.where(decoded_parts != self.tokenizer.mask_token)[0]

        for iter_idx in range(self.no_misspell):
            part_idx = random.choice(texts_ids)
            decoded_parts[part_idx] = make_typo(decoded_parts[part_idx])

        misspelled_masked_tokens = np.array(
            list(
                itertools.chain.from_iterable(
                    [part] if part in self.tokenizer.special_tokens_map.values() else self.tokenizer.tokenize(part)
                    for part in decoded_parts
                )
            )
        )

        misspelled_masked_tokens_mask_ids = np.where(misspelled_masked_tokens == self.tokenizer.mask_token)[0]
        misspelled_masked_tokens_ids = np.array(self.tokenizer.convert_tokens_to_ids(*misspelled_masked_tokens))

        return dict(
            **sample,
            misspelled_masked_tokens=misspelled_masked_tokens,
            misspelled_masked_tokens_ids=misspelled_masked_tokens_ids,
            misspelled_masked_tokens_mask_ids=misspelled_masked_tokens_mask_ids,
        )


class LabelCreatorTransform(BaseMisspellTransform):
    def __call__(self, sample: Dict[str, np.ndarray]):
        labels = np.full(len(sample["misspelled_masked_tokens"]), -100, dtype=np.int)
        labels[sample["misspelled_masked_tokens_mask_ids"]] = [
            self.tokenizer.convert_tokens_to_ids(token) for token in sample["tokens"][sample["mask"]]
        ]
        return dict(**sample, labels=labels)


class ConvertToIdsTransform:
    def __call__(self, sample: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(sample["misspelled_masked_tokens_ids"]),
            torch.from_numpy(sample["labels"]),
        )
