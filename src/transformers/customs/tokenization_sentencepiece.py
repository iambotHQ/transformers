from pathlib import Path
from typing import List, Optional, Sequence, Union
from overrides import overrides
from sentencepiece import SentencePieceProcessor
import torch
from typing import Any, Dict
from transformers.tokenization_utils import PreTrainedTokenizer

from transformers.tokenization_roberta import RobertaTokenizer

from lm import END_OF_LINE, END_OF_TEXT


class SentencePieceTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        model_path: Path,
        unk_token="<unk>",
        eot_token="</s>",
        cls_token="<cls>",
        sep_token="</s>",
        mask_token="<mask>",
        pad_token="<pad>",
        eos_token="</s>",
        bos_token="<s>",
        sampling: bool = False,
        **kwargs,
    ):
        self.sp = self.load_sentencepieceprocessor(model_path)
        self._eot_token = eot_token
        super(SentencePieceTokenizer, self).__init__(
            unk_token=unk_token, sep_token=sep_token, pad_token=pad_token, cls_token=cls_token, mask_token=mask_token, eos_token=eos_token, bos_token=bos_token, **kwargs,
        )
        self.max_len_single_sentence = self.max_len
        self.sampling = sampling 

    def __len__(self):
        return self.vocab_size

    @property
    def eot_token(self) -> str:
        return self._eot_token

    @property
    def eot_token_id(self) -> int:
        return self._convert_token_to_id(self.eot_token)

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

    def _convert_id_to_token(self, index: int) -> str:
        return self.sp.id_to_piece(index)

    def convert_ids_to_tokens(self, ids: Union[torch.Tensor, Sequence[int], int], skip_special_tokens: bool = False,) -> List[str]:
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        elif torch.is_tensor(ids):
            ids = ids.tolist()
        return [self._convert_id_to_token(index) for index in ids if not (skip_special_tokens and index in self.all_special_ids)]

    def _tokenize(self, text: str, add_prefix_space: bool = False) -> List[str]:  # type: ignore
        if add_prefix_space:
            text = " " + text
        return self.sp.sample_encode_as_pieces(text, -1, 0.1) if self.sampling else self.sp.encode_as_pieces(text)

    def convert_tokens_to_string(self, tokens: List[str], **kwargs: Dict[str, Any]) -> str:
        return "".join(tokens).replace("▁", " ").strip()

    @classmethod
    def _print_token(cls, token: str) -> None:
        print(token, end="", flush=True)

    def print_tokens(self, ids: List[int], token_id: int, print_token: bool = True) -> str:
        tokens, token = (
            self.convert_ids_to_tokens(ids),
            self._convert_id_to_token(token_id),
        )
        ending_puncts = "?!)])>:;}.,"
        starting_puncts = "([{<"

        normalized_token = token.replace(self.eos_token, "\n").replace(self.eot_token, "\n").replace("▁", " ")

        if (len(normalized_token) > 1 and normalized_token[1] in ending_puncts) or (len(tokens) > 1 and tokens[-2].replace("▁", "") in starting_puncts):
            normalized_token = normalized_token.replace(" ", "")

        if print_token:
            self._print_token(normalized_token)

        return normalized_token

    @property
    def stop_id(self) -> int:
        return self.sp.PieceToId(self.eot_token)

    _convert_token_to_id_with_added_voc = _convert_token_to_id
    build_inputs_with_special_tokens = RobertaTokenizer.build_inputs_with_special_tokens
    get_special_tokens_mask = RobertaTokenizer.get_special_tokens_mask
    create_token_type_ids_from_sequences = RobertaTokenizer.create_token_type_ids_from_sequences
    save_pretrained = RobertaTokenizer.save_pretrained
