from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from copy import deepcopy
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from allennlp.modules import FeedForward, Maxout
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.nn import util
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, MSELoss

from transformers.configuration_roberta import RobertaConfig
from transformers.customs.label_smoothing_loss import LabelSmoothingLoss
from transformers.file_utils import add_start_docstrings
from transformers.modeling_bert import (
    BertEmbeddings,
    BertLayerNorm,
    BertModel,
    BertPreTrainedModel,
    gelu,
)
from transformers.modeling_roberta import RobertaModel

logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


class RobertaForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, loss_function_type=torch.nn.functional.cross_entropy, **loss_function_kwargs):
        super(RobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)

        self.loss_function = partial(loss_function_type, **loss_function_kwargs)

    def forward(
        self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, hidden=None,
    ):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds,)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, hidden=hidden, text_mask=attention_mask)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss = self.loss_function(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_func = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.act_func(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaClassificationHeadBiLstm(nn.Module):
    dir_num: int = 2

    def __init__(self, config, device: torch.device = torch.device("cpu")):
        super(RobertaClassificationHeadBiLstm, self).__init__()
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=config.bilstm_hidden_size, bidirectional=self.dir_num == 2, batch_first=True, num_layers=config.bilstm_num_layers, dropout=config.bilstm_dropout,)
        self.dropout = torch.nn.AlphaDropout()
        self.hidden2label = nn.Linear(config.bilstm_hidden_size * self.dir_num, config.num_labels)

    @classmethod
    def init_hidden(cls, hidden_size: int, num_layers: int, batch_size: int, bidir: bool = True):
        return (
            Variable(torch.nn.init.xavier_uniform_(torch.Tensor(cls.dir_num * num_layers, batch_size, hidden_size).type(torch.FloatTensor)), requires_grad=True,),
            Variable(torch.nn.init.xavier_uniform_(torch.Tensor(cls.dir_num * num_layers, batch_size, hidden_size).type(torch.FloatTensor)), requires_grad=True,),
        )

    def forward(self, input, hidden, **kwargs):
        output, _ = self.lstm(input, hidden)
        out = self.hidden2label(output[:, -1])
        return out


class BCN(nn.Module):
    def __init__(self, config):
        super(BCN, self).__init__()

        # Pre-encode feed-forward
        self._embedding_dropout = nn.Dropout(0.25)
        self._pre_encode_feedforward = FeedForward(input_dim=config.hidden_size, num_layers=1, hidden_dims=[300], activations=[torch.nn.functional.relu], dropout=[0.25])

        # Encoder
        self._encoder = PytorchSeq2SeqWrapper(nn.LSTM(input_size=300, hidden_size=300, num_layers=1, bidirectional=True, batch_first=True))

        # Integrator
        self._integrator = PytorchSeq2SeqWrapper(nn.LSTM(input_size=1800, hidden_size=300, num_layers=1, bidirectional=True, batch_first=True))
        self._integrator_dropout = nn.Dropout(0.1)

        # Output layer

        self._self_attentive_pooling_projection = nn.Linear(self._integrator.get_output_dim(), 1)
        self._output_layer = Maxout(input_dim=2400, num_layers=3, output_dims=[1200, 600, config.num_labels], pool_sizes=4, dropout=[0.2, 0.3, 0.0])

    def forward(self, input, hidden, text_mask, **kwargs):
        text_mask = text_mask.type(torch.bool)

        dropped_embedded_text = self._embedding_dropout(input)  # [bs, seq_len, emb_size]
        pre_encoded_text = self._pre_encode_feedforward(dropped_embedded_text)  # [bs, seq_len, 300]
        encoded_tokens = self._encoder(pre_encoded_text, text_mask)  # [bs, seq_len, 600]

        # Bi attention
        attention_logits = encoded_tokens.bmm(encoded_tokens.permute(0, 2, 1).contiguous())  # [bs, seq_len, seq_len]
        attention_weights = util.masked_softmax(attention_logits, text_mask)  # [bs, seq_len, seq_len]
        encoded_text = util.weighted_sum(encoded_tokens, attention_weights)  # [bs, seq_len, 600]

        # Build the input to the integrator
        integrator_input = torch.cat([encoded_tokens, encoded_tokens - encoded_text, encoded_tokens * encoded_text], dim=2)  # [bs, seq_len, 1800]
        integrated_encodings = self._integrator(integrator_input, text_mask)  # [bs, seq_len, 600]

        # Concatenate LM representations
        # integrated_encodings = torch.cat([integrated_encodings, input], dim=-1)

        # Simple pooling layers
        max_maksed_integrated_encodings = util.replace_masked_values(integrated_encodings, text_mask.unsqueeze(2), -1e7)  # [bs, seq_len, 600]
        min_masked_integrated_encodings = util.replace_masked_values(integrated_encodings, text_mask.unsqueeze(2), +1e7)  # [bs, seq_len, 600]

        max_pool = torch.max(max_maksed_integrated_encodings, 1)[0]  # [bs, 600]
        min_pool = torch.min(min_masked_integrated_encodings, 1)[0]  # [bs, 600]
        mean_pool = torch.sum(integrated_encodings, 1) / torch.sum(text_mask, 1, keepdim=True)  # [bs, 600]

        # Self-attentive pooling layer
        self_attentive_logits = self._self_attentive_pooling_projection(integrated_encodings).squeeze(2)  # [bs, seq_len]
        self_weights = util.masked_softmax(self_attentive_logits, text_mask)  # [bs, seq_len]
        self_attentive_pool = util.weighted_sum(integrated_encodings, self_weights)  # [bs, 600]

        pooled_representations = torch.cat([max_pool, min_pool, mean_pool, self_attentive_pool], 1)  # [bs, 2400]
        pooled_representations_dropped = self._integrator_dropout(pooled_representations)  # [bs, 2400]

        return self._output_layer(pooled_representations_dropped)
