from pathlib import Path
from typing import List, Optional, Sequence, Union

from sentencepiece import SentencePieceProcessor
import torch

from transformers.tokenization_utils import PreTrainedTokenizer


class SentencePieceTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        model_path: Path,
        unk_token="<unk>",
        eot_token="<eot>",
        cls_token="<cls>",
        sep_token="<sep>",
        mask_token="<mask>",
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
        **kwargs,
    ):
        self.sp = self.load_sentencepieceprocessor(model_path)
        self.eot_token = eot_token
        super(SentencePieceTokenizer, self).__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            eos_token=eos_token,
            bos_token=bos_token,
            **kwargs,
        )
        self.max_len_single_sentence = self.max_len

    @property
    def __len__(self):
        return self.vocab_size

    @property
    def vocab_size(self):
        return len(self.sp)

    @classmethod
    def from_pretrained(cls, path: Path, **kwargs):  # type: ignore
        return cls(path, **kwargs)

    @classmethod
    def load_sentencepieceprocessor(cls, model_path: Path):
        sp = SentencePieceProcessor()
        sp.Load(f"{model_path}/sp.model")
        return sp

    def _convert_token_to_id(self, token: str) -> int:
        return self.sp.piece_to_id(token)

    _convert_token_to_id_with_added_voc = _convert_token_to_id

    def _convert_id_to_token(self, index: int) -> str:
        return self.sp.id_to_piece(index)

    def convert_ids_to_tokens(
        self, ids: Union[torch.Tensor, Sequence[int], int], skip_special_tokens: bool = False
    ) -> List[str]:
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        elif torch.is_tensor(ids):
            ids = ids.tolist()
        return [
            self._convert_id_to_token(index)
            for index in ids
            if not (skip_special_tokens and index in self.all_special_ids)
        ]

    def _tokenize(self, text: str, add_prefix_space: bool = False) -> List[str]:  # type: ignore
        if add_prefix_space:
            text = " " + text
        return self.sp.encode_as_pieces(text)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.sp.decode_pieces(tokens)

    @classmethod
    def _print_token(cls, token: str) -> None:
        print(token, end="", flush=True)

    def print_tokens(self, ids: List[int], token_id: int, print_token: bool = True) -> str:
        tokens, token = self.convert_ids_to_tokens(ids), self._convert_id_to_token(token_id)
        ending_puncts = "?!)])>:;}.,"
        starting_puncts = "([{<"

        normalized_token = (
            token.replace(self.eos_token, "\n").replace(self.eot_token, "\n").replace("▁", " ")
        )

        if (len(normalized_token) > 1 and normalized_token[1] in ending_puncts) or (
            len(tokens) > 1 and tokens[-2].replace("▁", "") in starting_puncts
        ):
            normalized_token = normalized_token.replace(" ", "")

        if print_token:
            self._print_token(normalized_token)

        return normalized_token

    @property
    def stop_id(self) -> int:
        return self.sp.PieceToId(self.eot_token)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]

        cls_token = [self.cls_token_id]
        sep_token = [self.sep_token_id]
        return cls_token + token_ids_0 + sep_token + sep_token + token_ids_1 + sep_token

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formated with special tokens for the model."
                )
            return list(
                map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0)
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> int:
        sep_token = [self.sep_token_id]
        cls_token = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls_token + token_ids_0 + sep_token) * [0]
        return len(cls_token + token_ids_0 + sep_token + sep_token + token_ids_1 + sep_token) * [0]
