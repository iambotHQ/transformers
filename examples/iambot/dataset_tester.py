import logging
import random
import re
import signal
import time
from argparse import Namespace
from contextlib import contextmanager
from copy import copy
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, Union, cast

import ipdb
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
from fire import Fire
from logzero import logger, loglevel
from mem_top import mem_top
from misspell import make_typo
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm import trange
from tqdm.auto import tqdm
from unidecode import unidecode

from pympler.tracker import SummaryTracker
from train_lm import IamBotLineByLineTextDataset
from transformers import PreTrainedTokenizer
from transformers.customs import SentencePieceTokenizer, wc


loglevel(logging.INFO)

special_tokens: Dict[str, str] = dict(
    unk_token="<unk>", eot_token="<eot>", sep_token="</s>", cls_token="<s>", mask_token="<mask>", pad_token="<pad>", eos_token="<eos>", bos_token="<bos>",
)


def create_dataloader(tokenizer: PreTrainedTokenizer, dataset: Dataset, transforms: bool, **kwargs: Any) -> DataLoader:
    def collate_with_masks(examples: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, masks = zip(*examples)
        masks = pad_sequence(masks, batch_first=True, padding_value=True)
        if tokenizer._pad_token is None:
            return pad_sequence(inputs, batch_first=True), masks
        return (
            pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id),
            masks,
        )

    def collate(examples: List[torch.Tensor]) -> torch.Tensor:
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    # logger.debug("Creating sampler")
    # sampler = RandomSampler(dataset)
    logger.debug("Creating dataloader")
    return DataLoader(dataset, collate_fn=collate_with_masks if transforms else collate, **kwargs)


def test_dataloader(
    tokenizer: PreTrainedTokenizer,
    transforms: bool,
    kwargs: Dict[str, Any],
    block_size: int,
    line_gen: Iterable[str],
    max_lines: int,
    batch_size: int,
    use_cuda: bool,
    repeats: int,
):
    last_one: bool = False

    if transforms:
        logger.warning("Will use transform mode")
        args: Namespace = Namespace(
            iambot_transform_ratio=1.0,
            iambot_misspell_prob=0.25,
            iambot_uppercase_prob=0.25,
            iambot_lowercase_prob=0.25,
            iambot_remove_char_prob=0.25,
            iambot_unidecode_prob=0.5,
            iambot_train_eval_ratio=0.03,
            block_size=block_size,
        )
    else:
        logger.warning("Will use vanilla mode")
        args = Namespace(**kwargs, block_size=block_size)

    while True:
        lines: List[str] = []
        while len(lines) <= max_lines if max_lines else True:
            try:
                lines.append(next(line_gen))
            except:
                last_one = True
                break

        dataset: Dataset = IamBotLineByLineTextDataset(tokenizer, args, None, transforms=transforms, lines=lines)


        for _ in trange(repeats, desc="Repeat"):
            dataloader: DataLoader = create_dataloader(
                tokenizer,
                dataset,
                transforms,
                batch_size=batch_size,
                # num_workers=0,
                num_workers=4,
                # timeout=5,
            )
            for batch in tqdm(dataloader, "Unpack data loader", leave=False):

                if transforms:
                    if use_cuda:
                        batch[0].cuda(), masks[0].cuda()
                    assert batch[0].shape[0] <= batch_size and batch[0].shape[1] < block_size, (batch[0].shape, batch[1].shape)
                    assert batch[1].shape[0] <= batch_size and batch[1].shape[1] < block_size, (batch[0].shape, batch[1].shape)
                else:
                    if use_cuda:
                        batch.cuda()
                    assert batch.shape[0] <= batch_size and batch.shape[1] < block_size, batch.shape
                import ipdb; ipdb.set_trace()
        if last_one:
            break


def main_test(
    dataset_path: Union[Path, str],
    tokenizer_path: Union[Path, str],
    transforms: bool = False,
    batch_size: int = 8,
    max_lines: int = 0,
    block_size: int = 512,
    sampling: bool = False,
    use_cuda: bool = False,
    repeats: int = 1,
    **kwargs: Any,
) -> None:
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer: PreTrainedTokenizer = SentencePieceTokenizer.from_pretrained(tokenizer_path, **special_tokens, max_len=block_size, sampling=sampling)
    line_gen: Iterable[str] = IamBotLineByLineTextDataset._read_lines(dataset_path)
    test_dataloader(tokenizer, transforms, kwargs, block_size, line_gen, max_lines, batch_size, use_cuda, repeats)


if __name__ == "__main__":
    Fire(main_test)
