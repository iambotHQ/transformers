import logging
import math
import multiprocessing as mp
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, cast

import torch
from logzero import logger
from misspell import make_typo
from tensorboardX.writer import SummaryWriter
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from unidecode import unidecode

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.customs import timeout, wc
from transformers.trainer import torch_distributed_zero_first

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def mp_tokenize_worker(tokenizer_creator: Callable[..., PreTrainedTokenizer], args, tokenized_lines_queue: mp.Queue, lines_queue: mp.Queue):
    # TODO
    tokenizer: PreTrainedTokenizer = tokenizer_creator()
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
        while True:
            try:
                _lines.extend(tokenized_queue.get(wait, 2))
            except:
                break
            if once:
                break

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


def iambot_train_eval(args, local_rank: int, tokenizer: PreTrainedTokenizer) -> None:
    # TODO
    with torch_distributed_zero_first(local_rank):
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


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."},
    )
    model_type: Optional[str] = field(
        default=None, metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    eval_data_file: Optional[str] = field(
        default=None, metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False, metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."})
    mlm_probability: float = field(default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"})

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached training and evaluation sets"})


@dataclass
class IamBotArgs:
    """
    Arguments for IamBot LM training modifications
    """

    iambot_mode: bool = field(default=False, metadata=dict(help="Enable IamBot mode"))
    iambot_tokenizer_sampling: bool = field(default=False, metadata=dict(help="Use tokenizer tokens sampling mode"))
    iambot_transforms: bool = field(default=False, metadata=dict(help="Train misspelled LM"))
    iambot_all_data: Optional[str] = field(default=None, metadata=dict(help="All data needed to train tokenizer and model"))
    iambot_vocab_size: int = field(default=20000, metadata=dict(help="Tokenizer vocab size"))
    iambot_force_train_tokenizer: bool = field(default=False, metadata=dict(help="Don't train tokenizer if already exists."))
    iambot_train_eval_ratio: float = field(default=0.05, metadata=dict(help="Split ratio of train and eval data."))
    iambot_force_split: bool = field(default=False, metadata=dict(help="Whether to force train/text split"))
    iambot_save_every_epoch: bool = field(default=False, metadata=dict(help="Whether to do checkpoint every epoch"))
    iambot_transform_ratio: float = field(default=0.2, metadata=dict(help="Ratio of transformed words in sentence"))
    iambot_misspell_prob: float = field(default=0.25, metadata=dict(help="Probability of misspell"))
    iambot_uppercase_prob: float = field(default=0.25, metadata=dict(help="Probability of uppercase"))
    iambot_lowercase_prob: float = field(default=0.25, metadata=dict(help="Probability of lowercase"))
    iambot_remove_char_prob: float = field(default=0.25, metadata=dict(help="Probability of remove char"))
    iambot_unidecode_prob: float = field(default=0.50, metadata=dict(help="Probability of unidecode"))


class IamBotUtils:
    tokenizers_special_tokens: Dict[str, Union[str, bool]] = dict(
        unk_token="<unk>", eot_token="<eot>", sep_token="</s>", cls_token="<s>", mask_token="<mask>", pad_token="<pad>", eos_token="<eos>", bos_token="<bos>",
    )

    @classmethod
    def create_iambot_config(
        cls,
        config: PretrainedConfig,
        tokenizer: PreTrainedTokenizer,
        training_args: TrainingArguments,
        model_args: ModelArguments,
        iambot_args: IamBotArgs,
        data_args: DataTrainingArguments,
    ) -> Tuple[PretrainedConfig, PreTrainedTokenizer]:
        logger.info("Creating SentencePieceTokenizer config")
        tokenizer_args: Dict[str, Union[str, bool]] = cls.tokenizers_special_tokens.copy()
        tokenizer_args["sampling"] = iambot_args.iambot_tokenizer_sampling and not training_args.do_eval

        logger.info(f"Modyfing {config.model_type} config")
        for attr_name, attr_val in dict(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            sep_token_id=tokenizer.sep_token_id,
            unk_token_id=tokenizer.unk_token_id,
            mask_token_id=tokenizer.mask_token_id,
            cls_token_id=tokenizer.cls_token_id,
        ).items():
            setattr(config, attr_name, attr_val)
        return config, tokenizer


class IamBotLineByLineTextDataset(Dataset):
    transforms: Tuple[str, str, str, str, str] = ("misspell", "uppercase", "lowercase", "remove_char", "unidecode")

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        block_size: int,
        args: IamBotArgs,
        file_path: Optional[Union[str, Path]] = None,
        lines: Optional[List[str]] = None,
        **kwargs: Any,
    ):

        self.tokenizer = tokenizer
        self.block_size = block_size
        self.args = args

        if lines:
            self.examples = lines
        elif file_path:
            file_path = Path(file_path)
            assert file_path.exists()
            logger.info(f"Loading lines from {file_path}")
            self.examples = list(self._read_lines(file_path))
        else:
            raise ValueError("You have to pass `lines` or `file_path`")

        self._input_length = self.block_size - 3
        self._features_creator: Callable[[str], Union[torch.Tensor, Union[torch.Tensor, torch.Tensor]]]
        if args.iambot_transforms:
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
        final_tokens, final_mask = cast(
            Tuple[Tuple[str, ...], Tuple[bool, ...]],
            zip(
                *(
                    (tok, mask)
                    for idx in range(len(tokens))
                    for tok, mask in zip(
                        *(
                            (tokenized_transformed_tokens[tokens_ids_to_transform.index(idx)], masks[tokens_ids_to_transform.index(idx)])
                            if idx in tokens_ids_to_transform
                            else ([tokens[idx]], [False])
                        )
                    )
                )
            ),
        )

        # build inputs for model
        final_tokens = self.tokenizer.build_inputs_with_special_tokens(self.tokenizer.convert_tokens_to_ids(final_tokens)[: self._input_length])
        # create final mask of maskable tokens
        final_mask = [True] + list(final_mask[: self._input_length]) + [True]

        return torch.tensor(final_tokens, dtype=torch.long), torch.tensor(final_mask, dtype=torch.bool)

    def create_features(self, text: str) -> torch.Tensor:
        tokens: List[str] = self.tokenizer.tokenize(text)[: self._input_length]
        ids: List[int] = cast(List[int], self.tokenizer.convert_tokens_to_ids(tokens))
        ids = self.tokenizer.build_inputs_with_special_tokens(ids)
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self._features_creator(self.examples[idx])


def get_dataset(args: DataTrainingArguments, iambot_args: IamBotArgs, tokenizer: PreTrainedTokenizer, evaluate=False, local_rank=-1):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.line_by_line:
        if iambot_args.iambot_mode:
            return IamBotLineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, local_rank=local_rank, args=iambot_args)
        return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, local_rank=local_rank)
    else:
        return TextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, local_rank=local_rank,)


@dataclass
class IamBotDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def collate_batch(self, examples: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_inputs, batch_tokens_masks = cast(Tuple[List[torch.Tensor], List[torch.Tensor]], zip(*examples))
        inputs, tokens_mask = self._tensorize_batch(batch_inputs), self._tensorize_batch(batch_tokens_masks, True)
        if self.mlm:
            inputs, labels = self.mask_tokens(inputs, tokens_mask)
            return {"input_ids": inputs, "masked_lm_labels": labels}
        else:
            return {"input_ids": inputs, "labels": inputs}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, IamBotArgs))
    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments
    iambot_args: IamBotArgs
    model_args, data_args, training_args, iambot_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file " "or remove the --do_eval argument.")

    if os.path.exists(training_args.output_dir) and os.listdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if iambot_args.iambot_mode:
        IamBotUtils.create_iambot_config(config, tokenizer, training_args, model_args, iambot_args, data_args)

    if model_args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            model_args.model_name_or_path, from_tf=bool(".ckpt" in model_args.model_name_or_path), config=config, cache_dir=model_args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if config.model_type in ["bert", "roberta", "distilbert", "camembert"] and not data_args.mlm:
        raise ValueError("BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm " "flag (masked language modeling).")

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    # Get datasets
    # TODO iambot split
    train_dataset = get_dataset(data_args, iambot_args, tokenizer=tokenizer, local_rank=training_args.local_rank) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, iambot_args, tokenizer=tokenizer, local_rank=training_args.local_rank, evaluate=True) if training_args.do_eval else None
    data_collator_class = IamBotDataCollatorForLanguageModeling if iambot_args.iambot_mode else DataCollatorForLanguageModeling
    data_collator = data_collator_class(tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        prediction_loss_only=True,
        compute_perplexity=True,
    )

    # Training
    if training_args.do_train:
        model_path = model_args.model_name_or_path if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path) else None
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
