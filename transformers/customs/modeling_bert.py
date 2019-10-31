from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as scatter
from itertools import starmap
from transformers.configuration_bert import BertConfig
from transformers.modeling_bert import BertModel, BertPreTrainedModel


class SimpleConcatAvgMaxTokensPooler(nn.Module):
    def __init__(self, config):
        super(SimpleConcatAvgMaxTokensPooler, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, tokens_embs: torch.Tensor, *args) -> torch.Tensor:  # type: ignore
        return torch.cat(
            [self.avg_pool(tokens_embs).squeeze(), self.max_pool(tokens_embs).squeeze()], dim=0
        ).squeeze()


class SimpleAvgOrMaxTokensPoolerWithMask(nn.Module):
    def __init__(self, config):
        super(SimpleAvgOrMaxTokensPoolerWithMask, self).__init__()
        word_tokens_pooling_method = (
            getattr(config, "word_tokens_pooling_method", "").lower().capitalize()
        )
        self.pooler = getattr(nn, f"Adaptive{word_tokens_pooling_method}Pool1d")(1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.pooler(tensor).squeeze()


class CustomBertForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(CustomBertForNer, self).__init__(config)
        word_tokens_pooling_method = getattr(config, "word_tokens_pooling_method", "").lower()
        linear_hidden_size_mult = 1

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if word_tokens_pooling_method in ["avg", "max"]:
            self.tokens_pooler = SimpleAvgOrMaxTokensPoolerWithMask(config)
        elif word_tokens_pooling_method == "concatavgmax":
            self.tokens_pooler = SimpleConcatAvgMaxTokensPooler(config)
            linear_hidden_size_mult = 2

        self.classifier = nn.Linear(config.hidden_size * linear_hidden_size_mult, config.num_labels)

        self.init_weights()

    def _convert_bert_outputs_to_map(
        self, outputs: Tuple[torch.Tensor, ...]
    ) -> Dict[str, torch.Tensor]:
        outputs_map = dict(last_hidden_state=outputs[0], pooler_output=outputs[1])
        if len(outputs) > 2:
            outputs_map["hidden_states"] = outputs[2]
        if len(outputs) > 3:
            outputs_map["attentions"] = outputs[3]
        return outputs_map


    def forward(
        self,
        input_ids,
        word_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )
        outputs = self._convert_bert_outputs_to_map(outputs)
        sequence_output = outputs["last_hidden_state"]

        if word_ids is not None and hasattr(self, 'tokens_pooler'):

            word_ids[word_ids == -100] = -1

            _word_ids = word_ids.unsqueeze(-1) + 1
            _mean = scatter.scatter_mean(sequence_output, _word_ids, dim=1).type(sequence_output.dtype)
            _mean[:, 0, :] = sequence_output[:, 0, :]
            _max = scatter.scatter_max(sequence_output, _word_ids, dim=1, fill_value=0)[0].type(sequence_output.dtype)
            _max[:, 0, :] = sequence_output[:, 0, :]
            sequence_output = torch.cat([_mean, _max], dim=-1)

            word_ids[word_ids == -1] = -100

            def transform_ids(word_ids: torch.Tensor, labels: torch.Tensor, pad_id: int = -100) -> torch.Tensor:
                word_labels = labels[word_ids[word_ids != pad_id].unique_consecutive(return_counts=True)[1].cumsum(dim=0) - 1]
                tensor = F.pad(word_labels, (0, sequence_output.shape[1] - 1 - word_labels.shape[0]), value=pad_id)
                return tensor

            labels = torch.stack(list(starmap(transform_ids, zip(word_ids[:, 1:], labels[:, 1:]))), dim=0)
            labels = torch.cat((torch.tensor(-100).repeat(labels.shape[0], 1).to(labels.device), labels), dim=1)
            attention_mask = torch.zeros_like(labels)
            attention_mask[labels != -100] = 1

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1).type(torch.bool)
                active_logits = logits.view(-1, self.config.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            outputs["loss"] = loss

        outputs["attention_mask"] = attention_mask
        outputs["logits"] = logits
        outputs["labels"] = labels

        return outputs  # (loss), scores, (hidden_states), (attentions)


class CustomBertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, attention_mask=None, num_question_tokens=None, position_ids=None, head_mask=None, start_positions=None, end_positions=None):
        batch_size, seq_len = input_ids.shape
        windowed_mode = False

        if seq_len > self.config.max_position_embeddings:
            assert batch_size == 1, 'sliding window mode is not currently supported for batch_size > 1'
            assert position_ids is None
            assert isinstance(num_question_tokens, int)

            windowed_mode = True

            input_ids = torch.squeeze(input_ids, dim=0)
            question_ids, paragraph_ids = torch.split(input_ids, [num_question_tokens, seq_len - num_question_tokens])
            windowed_paragraph_ids, paragraph_position_ids = self._apply_sliding_window_to_single_batch(
                paragraph_ids, self.config.max_position_embeddings - num_question_tokens)

            batch_size = windowed_paragraph_ids.shape[0]
            seq_len = self.config.max_position_embeddings

            input_ids = torch.cat(
                (question_ids.unsqueeze(0).expand(batch_size, num_question_tokens), windowed_paragraph_ids),
                dim=-1
            )

        if num_question_tokens is None:
            num_question_tokens = seq_len

        if isinstance(num_question_tokens, int):
            token_type_ids = self._create_type_tokens_for_single_batch(num_question_tokens, seq_len).unsqueeze(0).expand(batch_size, seq_len)
        else:
            token_type_ids = torch.stack([self._create_type_tokens_for_single_batch(num_in_batch, seq_len) for num_in_batch in num_question_tokens])

        token_type_ids = token_type_ids.to(input_ids.device)

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask)
        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        if windowed_mode:
            question_logits, paragraph_logits = torch.split(logits, [num_question_tokens, seq_len - num_question_tokens], dim=1)

            question_logits = question_logits.min(dim=0, keepdim=True)[0]
            paragraph_logits = self._compress_sliding_window(paragraph_logits, paragraph_position_ids)

            logits = torch.cat((question_logits, paragraph_logits), dim=1)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)

    def _create_type_tokens_for_single_batch(self, num_question_tokens, seq_len):
        return torch.cat([torch.zeros(num_question_tokens), torch.ones(seq_len - num_question_tokens)]).long()

    def _apply_sliding_window_to_single_batch(self, tokens, window_size=512, window_stride=None):
        if window_stride is None:
            window_stride = window_size // 2

        result_batch = []
        result_positions = []

        start = end = 0
        while end < len(tokens):
            end = min(start + window_size, len(tokens))
            start = end - window_size # this allows to avoid shorter last window

            result_batch.append(tokens[start:end])
            result_positions.append(torch.arange(start, end))

            start += window_stride

        return torch.stack(result_batch), torch.stack(result_positions)

    def _compress_sliding_window(self, windowed_tokens, positions):
        num_tokens = positions.max() + 1
        window_size = windowed_tokens.shape[1]

        middle_positions = positions[:, window_size // 2]
        tokens_best_window_idxs = (torch.unsqueeze(middle_positions, 0) - torch.arange(num_tokens).unsqueeze(-1)).abs().argmin(dim=1)
        token_mask = torch.stack(
            [tokens_best_window_idxs[token_idxs_in_window] == window_idx for window_idx, token_idxs_in_window in enumerate(positions)]
        )

        return windowed_tokens[token_mask].unsqueeze(0)
