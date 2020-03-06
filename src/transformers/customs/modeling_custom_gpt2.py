from pathlib import Path
from transformers.customs import override
from lm.inference import ModelWrapper
from lm.model import OutputGetters

from transformers.configuration_gpt2 import GPT2Config
from transformers.modeling_gpt2 import GPT2LMHeadModel


class CustomGPT2(GPT2LMHeadModel):
    config_class = GPT2Config
    config = GPT2Config()

    def __init__(self, model: ModelWrapper):
        super(CustomGPT2, self).__init__(self.config)
        self.transformer = model

    @override
    @classmethod
    def from_pretrained(cls, path: Path, generation_mode: bool = False, *args, **kwargs):
        model = ModelWrapper.load_encoder(
            f"{path}/model.pt", True, True, "cpu", output_getter=OutputGetters.raw
        )
        model.config = cls.config
        setattr(model.config, 'generation_mode', True)
        setattr(model.config, 'output_past', False)
        setattr(model.config, 'vocab_size', model.hparams.n_vocab)
        setattr(model.config, 'n_ctx', model.hparams.n_ctx)
        setattr(model.config, 'n_embd', model.hparams.n_embed)
        setattr(model.config, 'n_layer', model.hparams.n_layer)
        setattr(model.config, 'n_head', model.hparams.n_head)
        return cls(model).to_generation_mode()

    def to_generation_mode(self):
        if getattr(self.config, "generation_mode", False):
            setattr(self, "forward", self._generation_call)
        return self

    @property
    def base_model(self) -> ModelWrapper:
        return self.transformer

    @override
    def get_output_embeddings(self):
        # Because transformer-lm doens't have any lm_head, like huggingface transformers
        return self

    def eval(self):
        self.transformer.eval()

    def to(self, device):
        self.transformer = self.transformer.to(device)

    def _generation_call(self, input_ids, *args, **kwargs):
        outputs = self.transformer(input_ids)
        return outputs["logits"], outputs["presents"]

    def forward(self, input_ids, *args, **kwargs):
        output = self.transformer(input_ids)["logits"]
        return (output,)
