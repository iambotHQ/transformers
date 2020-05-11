import argparse
import gc
import glob
import pandas as pd
import logging
import multiprocessing as mp
import os
import pickle
import random
import re
import shutil
import signal
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast

import numpy as np
import torch
import torch.multiprocessing
from logzero import logfile, logger, loglevel
from misspell import make_typo
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from unidecode import unidecode

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    get_linear_schedule_with_warmup,
)
from transformers.customs import SentencePieceTokenizer, wc

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
import gc

loglevel(level=logging.DEBUG, update_custom_handlers=False)
logfile("./logs.txt", loglevel=logging.DEBUG)


def mem_report():
    """Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported"""

    def _mem_report(tensors, mem_type):
        """Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation """
        logger.warning("Storage on %s" % (mem_type))
        logger.warning("-" * LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel * element_size / 1024 / 1024  # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            # logger.warning("%s\t\t%s\t\t%.2f" % (element_type, size, mem))
        logger.warning("-" * LEN)
        logger.warning("Total Tensors: %d \tUsed Memory Space: %.2f MBytes" % (total_numel, total_mem))
        logger.warning("-" * LEN)

    LEN = 65
    logger.warning("=" * LEN)
    objects = gc.get_objects()
    logger.warning("%s\t%s\t\t\t%s" % ("Element type", "Size", "Used MEM(MBytes)"))
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, "GPU")
    _mem_report(host_tensors, "CPU")
    logger.warning("=" * LEN)


def raise_timeout(signum, frame):
    raise TimeoutError


@contextmanager
def timeout(time: int):
    signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(time)

    try:
        yield
    except TimeoutError:
        pass
    finally:
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str):
        assert os.path.isfile(file_path)

        block_size = args.block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename,)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in trange(0, len(tokenized_text) - block_size + 1, block_size, desc="Creating blocks"):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.strip() for line in tqdm(f, "Reading lines") if (len(line.strip()) > 0 and not line.isspace())]

        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=args.block_size)["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


class IamBotLineByLineTextDataset(Dataset):
    transforms: Tuple[str, str, str, str, str] = ("misspell", "uppercase", "lowercase", "remove_char", "unidecode")

    def __init__(self, tokenizer: PreTrainedTokenizer, args: Namespace, file_path: Optional[Union[str, Path]] = None, transforms: bool = False, lines: List[str] = []):

        self.tokenizer = tokenizer
        self.block_size = args.block_size
        self.args = args

        if lines:
            self.examples = lines
        elif file_path:
            file_path = Path(file_path)
            assert file_path.exists()
            logger.info(f"Creating features from dataset file {file_path}")
            self.examples = list(self._read_lines(file_path))
        else:
            raise ValueError("You have to pass `lines` or `file_path`")

        self._input_length = self.block_size - 3
        if transforms:
            logger.warning("Dataset with noise")
            self._features_creator = self.create_features_with_noise
        else:
            logger.warning("Dataset without noise")
            self._features_creator = self.create_features

    @classmethod
    def _read_lines(cls, file_path: Path) -> Iterable[str]:
        with open(file_path, "r") as fhd:
            yield from (line for line in (line.strip() for line in tqdm(fhd, f"Reading lines from {file_path}", total=wc(file_path))) if line)

    def __len__(self) -> int:
        return len(self.examples)

    @classmethod
    def _transform_with_prob(cls, text: str, transform: Callable[[str], str], prob: float) -> str:
        return transform(text) if random.uniform(0.0, 1.0) <= prob else text

    @classmethod
    def _transform_misspell(cls, text: str) -> str:
        with timeout(1):
            text = make_typo(text, unicode=True)
        return text

    @classmethod
    def _transform_uppercase(cls, text: str) -> str:
        return text.upper()

    @classmethod
    def _transform_lowercase(cls, text: str) -> str:
        return text.lower()

    @classmethod
    def _transform_remove_char(cls, text: str) -> str:
        char_to_remove = random.randrange(len(text))
        _text = list(text)
        del _text[char_to_remove]
        new_text = "".join(_text)
        return new_text

    @classmethod
    def _transform_unidecode(cls, text: str) -> str:
        return cast(str, unidecode(text))

    def transform(self, text: str) -> str:
        for transform_name in self.transforms:
            transformer: Callable[[str], str] = getattr(self, f"_transform_{transform_name}")
            prob: float = getattr(self.args, f"iambot_{transform_name}_prob")
            text = self._transform_with_prob(text, transformer, prob)
        return text

    def create_features_with_noise(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        # tokenize text
        tokens: List[str] = self.tokenizer.tokenize(text)
        # choose ids of tokens to be transformed
        tokens_ids_to_transform: List[int] = sorted(random.sample(range(len(tokens)), int(self.args.iambot_transform_ratio * len(tokens))))
        # extract tokens to be transformed
        tokens_to_transform: List[str] = [tokens[idx] for idx in tokens_ids_to_transform]
        # transform chosen tokens
        transformed_tokens: List[str] = [self.transform(token) for token in tokens_to_transform]
        # try to convert transformed tokens to ids
        transformed_tokens_ids: List[int] = [self.tokenizer._convert_token_to_id(token) for token in transformed_tokens]
        # tokenize transformed tokens if they're not valid
        tokenized_transformed_tokens: List[Union[List[str], str]] = [
            self.tokenizer.tokenize(token) if token_id == self.tokenizer.unk_token_id else [token]
            for token, token_id in zip(transformed_tokens, transformed_tokens_ids)
        ]

        # create mask of possible mask tokens
        masks: List[List[bool]] = [
            [trans_token not in orig_token for trans_token in trans_token_tokens]
            for orig_token, trans_token_tokens in zip(tokens_to_transform, tokenized_transformed_tokens)
        ]

        # unfold masks and tokens
        final_tokens, final_mask = zip(
            *(
                (tok, mask)
                for idx in range(len(tokens))
                for tok, mask in zip(
                    *(
                        (tokenized_transformed_tokens[tokens_ids_to_transform.index(idx)], masks[tokens_ids_to_transform.index(idx)],)
                        if idx in tokens_ids_to_transform
                        else ([tokens[idx]], [False])
                    )
                )
            )
        )

        # build inputs for model
        final_tokens = self.tokenizer.build_inputs_with_special_tokens(self.tokenizer.convert_tokens_to_ids(final_tokens)[: self._input_length])
        # create final mask of maskable tokens
        final_mask = [True] + list(final_mask)[: self._input_length] + [True]

        return torch.tensor(final_tokens, dtype=torch.long), torch.tensor(final_mask, dtype=torch.bool)

    def old_misspelled(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        sentence = pd.DataFrame(dict(words=list(filter(str.strip, re.split(r"\b", text.strip())))))
        sentence["trans_words"] = sentence["words"].copy()
        sentence["transformed"] = False

        to_transform = sentence.sample(frac=self.args.iambot_transform_ratio, replace=False)
        sentence.loc[to_transform.index, "trans_words"] = to_transform["trans_words"].apply(self.transform)
        sentence.loc[to_transform.index, "transformed"] = True

        new_text = " ".join(sentence["trans_words"]).strip()
        old_text = " ".join(sentence["words"]).strip()

        sentence["tokens"] = sentence["trans_words"].apply(self.tokenizer.tokenize)

        unfolded: List[List[Union[int, bool]]] = list(
            chain.from_iterable(
                [[token, originality or (set(original_words) == set(trans_words))] for token in tokens]
                for tokens, originality, original_words, trans_words in (
                    (self.tokenizer.convert_tokens_to_ids(data[0]), not data[1], data[2], data[3],)
                    for data in sentence[["tokens", "transformed", "words", "trans_words"]].itertuples(index=False)
                )
            )
        )

        _inputs, _maskable_tokens = cast(Tuple[List[int], List[bool]], list(map(list, zip(*unfolded))))
        inputs = torch.tensor(self.tokenizer.build_inputs_with_special_tokens(_inputs[: self.args.block_size - 3]), dtype=torch.long)
        maskable_tokens = torch.tensor([False, *(_maskable_tokens[: self.args.block_size - 3]), False], dtype=torch.bool)

        try:
            assert inputs.shape == maskable_tokens.shape and len(inputs) < self.tokenizer.max_len
        except Exception as e:
            ipdb.set_trace()

        return inputs, maskable_tokens

    def create_features(self, text: str) -> torch.Tensor:
        return torch.tensor(
            self.tokenizer.build_inputs_with_special_tokens(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text)[: self._input_length])), dtype=torch.long
        )

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._features_creator(self.examples[idx])


def load_and_cache_examples(args, tokenizer: PreTrainedTokenizer, evaluate: bool = False) -> Dataset:
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.iambot:
        return IamBotLineByLineTextDataset(tokenizer, args, file_path, args.iambot_transforms)
    else:
        dataset_class = LineByLineTextDataset if args.line_by_line else TextDataset
    return dataset_class(tokenizer, args, file_path)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix: str = "checkpoint", use_mtime: bool = False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def mask_tokens(
    inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], tokenizer: PreTrainedTokenizer, args, **kwargs: Any
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )
    masks: Optional[torch.Tensor] = None

    if len(inputs) == 2:
        inputs, masks = inputs

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if masks is not None:
        probability_matrix.masked_fill_(masks, value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def save_checkpoint(
    args, global_step: int, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, optimizer, scheduler,
):
    checkpoint_prefix = "checkpoint"
    # Save model checkpoint
    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    if not args.iambot:
        tokenizer.save_pretrained(output_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", output_dir)

    _rotate_checkpoints(args, checkpoint_prefix)

    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    logger.info("Saving optimizer and scheduler states to %s", output_dir)


def train(args, train_dataset: Dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(Path(args.output_dir) / "logs" / "train", flush_secs=10)
        tb_writer.add_hparams({k: v or 0 for k, v in vars(args).items() if k not in ["device"]}, {})

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

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

    # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        # sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate_with_masks if args.iambot_transforms else collate,
        num_workers=4,
        pin_memory=True,
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay,},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"), map_location=args.device,))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"), map_location=args.device,))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info(
                "  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch,
            )
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],)
    set_seed(args)  # Added here for reproducibility

    batch_modifier: Callable[[Tuple[torch.Tensor, torch.Tensor]], Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],] = lambda batch: (
        batch,
        batch,
    )
    model_call: Callable[..., Tuple[torch.Tensor, ...]] = lambda inputs, labels: model(inputs, labels=labels)
    loss_avger: Callable[[torch.Tensor], torch.Tensor] = lambda loss: loss
    loss_accumulator: Callable[[torch.Tensor], torch.Tensor] = lambda loss: loss

    if args.mlm:
        batch_modifier = lambda batch: mask_tokens(batch, tokenizer, args)
        model_call = lambda inputs, labels: model(inputs, masked_lm_labels=labels)

    if args.n_gpu > 1:
        loss_avger = lambda loss: loss.mean()

    if args.gradient_accumulation_steps > 1:
        loss_accumulator = lambda loss: loss / args.gradient_accumulation_steps

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            with torch.no_grad():
                inputs, labels = batch_modifier(batch)
                inputs, labels = inputs.to(args.device), labels.to(args.device)
                model.train()
                loss = model_call(inputs, labels)[0]
                loss = loss_avger(loss)
                loss = loss_accumulator(loss)

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)

                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    avg_loss = (tr_loss - logging_loss) / args.logging_steps
                    tb_writer.add_scalar("loss", avg_loss, global_step)
                    tb_writer.add_scalar("perplexity", torch.exp(torch.tensor(avg_loss)), global_step)
                    logging_loss = tr_loss

                if not args.iambot_save_every_epoch and args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(args, global_step, model, tokenizer, optimizer, scheduler)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.iambot_save_every_epoch and args.local_rank in [-1, 0]:
            save_checkpoint(args, global_step, model, tokenizer, optimizer, scheduler)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    eval_output_dir = Path(args.output_dir) / "logs" / "eval"

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    def collate_with_masks(examples: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs, masks = zip(*examples)
        masks = pad_sequence(masks, batch_first=True, padding_value=False)
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

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_with_masks if args.iambot_transforms else collate,
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    result = {"perplexity": perplexity, "loss": eval_loss}

    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))
    return result


special_tokens: Dict[str, str] = dict(
    unk_token="<unk>", eot_token="<eot>", sep_token="</s>", cls_token="<s>", mask_token="<mask>", pad_token="<pad>", eos_token="<eos>", bos_token="<bos>",
)


def iambot_tokenizer(args) -> Tuple[PreTrainedTokenizer, PretrainedConfig]:

    tokenizer_dir = Path(args.tokenizer_name)
    logger.info("Creating SentencePieceTokenizer")
    tokenizer = SentencePieceTokenizer(tokenizer_dir, **special_tokens, sampling=args.iambot_tokenizer_sampling and not args.do_eval)
    logger.info("Creating RobertaConfig")
    config = RobertaConfig(
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sep_token_id=tokenizer.sep_token_id,
        unk_token_id=tokenizer.unk_token_id,
        mask_token_id=tokenizer.mask_token_id,
        cls_token_id=tokenizer.cls_token_id,
        architectures=["RobertaForMaskedLM"],
    )
    if args.local_rank == -1 or args.no_cuda:
        config.save_pretrained(args.tokenizer_name)
    tokenizer.max_len = config.max_position_embeddings
    return tokenizer, config


def mp_tokenize_worker(args, tokenized_lines_queue: mp.Queue, lines_queue: mp.Queue):

    tokenizer: SentencePieceTokenizer = iambot_tokenizer(args)[0]
    while True:
        line: Optional[str] = lines_queue.get()
        if line is None:
            logger.warning(f"Tokenizer {mp.current_process().name:3}: stopping")
            lines_queue.task_done()
            break
        tokenized_lines_queue.put(tokenizer.tokenize(line))
        lines_queue.task_done()

    logger.warning(f"Tokenizer {mp.current_process().name:3}: finished")


def mp_tokenize(lines: List[str], args) -> List[str]:
    _lines: List[str] = []

    nproc: int = mp.cpu_count()
    processes: List[mp.Process] = []
    lines_queue: mp.JoinableQueue = mp.JoinableQueue()
    tokenized_queue: mp.Queue = mp.Queue()

    def gather(wait: bool = True, once: bool = False):
        counter: int = 0
        while True:
            try:
                _lines.extend(tokenized_queue.get(wait, 2))
            except:
                break
            if once:
                break
            counter += 1

    for idx in range(nproc):
        process: mp.Process = mp.Process(
            target=mp_tokenize_worker, args=(args, tokenized_queue, lines_queue), name=str(idx),
        )
        process.start()
        processes.append(process)

    for idx, line in enumerate(lines):
        lines_queue.put(line)
        if idx % 100 == 0 and not tokenized_queue.empty():
            gather(once=True)

    for _ in range(len(processes)):
        lines_queue.put(None)

    gather()
    logger.info(f"Joining lines queue")
    lines_queue.join()
    logger.info("Removing lines")
    del lines
    gather()
    return _lines


def tokenize_sp(lines: Iterable[str], tokenizer: PreTrainedTokenizer) -> List[str]:
    tokenized_text: List[List[str]] = [tokenizer.tokenize(sentence) for sentence in tqdm(lines, "Tokenizing")]
    no_all_tokens: int = sum(len(sentence) for sentence in tqdm(tokenized_text, "Counting tokens"))
    all_tokens: List[str] = [""] * no_all_tokens
    last_idx: int = 0
    for sentence in tqdm(tokenized_text, "Creating tokens", unit="sentence"):
        all_tokens[last_idx : last_idx + len(sentence)] = sentence
        last_idx += len(sentence)

    return all_tokens


def iambot_train_eval(args: Namespace, tokenizer: SentencePieceTokenizer) -> None:

    if not (Path(args.train_data_file).exists() and Path(args.eval_data_file)) or args.iambot_force_split:
        no_eval_tokens: int = 1_500_000

        def save_lines(lines: Iterable[str], path: str):
            with open(path, "w") as fhd:
                for line in tqdm(lines, f"Saving lines to {path}"):
                    fhd.write(f"{line}\n")

        with open(args.iambot_all_data, "r") as fhd:
            lines = [line for line in (line.strip() for line in tqdm(fhd, f"Loading lines from {args.iambot_all_data}", wc(args.iambot_all_data),)) if line]

        all_tokens: List[str] = mp_tokenize(lines, args)
        train_data: Iterable[str] = (
            tokenizer.convert_tokens_to_string(all_tokens[i : i + args.block_size - 2], skip_special_tokens=True)
            for i in trange(0, len(all_tokens) - no_eval_tokens, args.block_size - 2, desc="Creating training examples",)
        )

        eval_data: Iterable[str] = (
            tokenizer.convert_tokens_to_string(all_tokens[i : i + args.block_size - 2], skip_special_tokens=True)
            for i in trange(len(all_tokens) - no_eval_tokens, len(all_tokens) - args.block_size - 2, args.block_size - 2, desc="Creating eval examples",)
        )

        save_lines(train_data, args.train_data_file)
        save_lines(eval_data, args.eval_data_file)
        exit(0)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    ## IAMBOT
    parser.add_argument("--iambot", action="store_true", help="IamBot training mode")
    parser.add_argument("--iambot_tokenizer_sampling", action="store_true", help="Use tokenizer tokens sampling mode")
    parser.add_argument("--iambot_transforms", action="store_true", help="Train misspelled LM")
    parser.add_argument(
        "--iambot_all_data", type=str, help="All data needed to train tokenizer and model",
    )
    parser.add_argument("--iambot_vocab_size", default=25000, type=int, help="Tokenizer vocab size")
    parser.add_argument(
        "--iambot_force_train_tokenizer", default=False, type=bool, help="Don't train tokenizer if already exists.",
    )
    parser.add_argument(
        "--iambot_train_eval_ratio", default=0.05, type=float, help="Split ratio of train and eval data.",
    )
    parser.add_argument(
        "--iambot_force_split", default=False, action="store_true", help="Whether to force train/text split",
    )
    parser.add_argument(
        "--iambot_save_every_epoch", action="store_true", default=False, help="Whether to do checkpoint every epoch",
    )
    parser.add_argument(
        "--iambot_transform_ratio", type=float, default=0.2, help="Ratio of transformed words in sentence",
    )
    parser.add_argument(
        "--iambot_misspell_prob", type=float, default=0.25, help="Probability of misspell",
    )
    parser.add_argument(
        "--iambot_uppercase_prob", type=float, default=0.25, help="Probability of uppercase",
    )
    parser.add_argument(
        "--iambot_lowercase_prob", type=float, default=0.25, help="Probability of lowercase",
    )
    parser.add_argument(
        "--iambot_remove_char_prob", type=float, default=0.25, help="Probability of remove char",
    )
    parser.add_argument(
        "--iambot_unidecode_prob", type=float, default=0.50, help="Probability of unidecode",
    )
    parser.add_argument(
        "--eval_data_file", default=None, type=str, help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line", action="store_true", help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir",
    )
    parser.add_argument(
        "--model_name_or_path", default=None, type=str, help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling.",
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss",
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir", default=None, type=str, help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']." "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="For distributed training: local_rank",
    )
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()
    tokenizer: Optional[PreTrainedTokenizer] = None

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError("BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm " "flag (masked language modeling).")
    if args.eval_data_file is None and args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file " "or remove the --do_eval argument.")
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    if args.iambot:
        logger.info("IambBot mode")
        if args.iambot_transforms:
            logger.warning("IamBot: misspelling mode")
        else:
            logger.warning("IamBot: classic mode (without misspelling)")

        if args.model_type.lower() != "roberta":
            raise ValueError(f"IamBot currrently supports only roberta model. Set --model_type to `roberta`.")
        tokenizer, config = iambot_tokenizer(args)
    else:
        if args.config_name:
            config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
        elif args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            # When we release a pip version exposing CONFIG_MAPPING,
            # we can do `config = CONFIG_MAPPING[args.model_type]()`.
            raise ValueError(
                "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
                "and load it from here, using --config_name"
            )

        if args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
        elif args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
                "and load it from here, using --tokenizer_name"
            )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    assert args.block_size < 1000, "meh"

    if args.iambot and (args.local_rank in [-1, 0] or args.no_cuda):
        iambot_train_eval(args, tokenizer)

    if args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(args.model_name_or_path, from_tf=bool(".ckpt" in args.model_name_or_path), config=config, cache_dir=args.cache_dir,)
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        # tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = AutoModelWithLMHead.from_pretrained(args.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c)
                for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True), key=lambda path: int(path.split("/")[-2].split("-")[-1]),)
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        tb_writer = SummaryWriter(Path(args.output_dir) / "logs" / "eval", flush_secs=10)
        pbar = tqdm(checkpoints)
        for checkpoint in pbar:
            pbar.set_description(f"Checkpoint: {Path(checkpoint).name}")
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model = AutoModelWithLMHead.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            tb_writer.add_scalar("loss", result["loss"], global_step)
            tb_writer.add_scalar("perplexity", result["perplexity"], global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
