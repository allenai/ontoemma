from typing import Dict

import torch
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder


@TextFieldEmbedder.register("list_embedder")
class ListTextFieldEmbedder(TextFieldEmbedder):
    """
    Same as basic text field embedder but for lists
    """
    def __init__(self, token_embedders: Dict[str, TokenEmbedder]) -> None:
        super(ListTextFieldEmbedder, self).__init__()
        self._token_embedders = token_embedders
        for key, embedder in token_embedders.items():
            name = 'token_embedder_%s' % key
            self.add_module(name, embedder)

    @overrides
    def get_output_dim(self) -> int:
        output_dim = 0
        for embedder in self._token_embedders.values():
            output_dim += embedder.get_output_dim()
        return output_dim

    def forward(self, text_field_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self._token_embedders.keys() != text_field_input.keys():
            message = "Mismatched token keys: %s and %s" % (str(self._token_embedders.keys()),
                                                            str(text_field_input.keys()))
            raise ConfigurationError(message)
        embedded_representations = []
        keys = sorted(text_field_input.keys())

        for key in keys:
            tensor = text_field_input[key]
            embedder = self._token_embedders[key]
            list_representations = []
            for row in tensor:
                token_vectors = embedder(row)
                list_representations.append(token_vectors)
            embedded_representations.append(torch.stack(list_representations))

        return torch.cat(embedded_representations, dim=-1)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'ListTextFieldEmbedder':
        token_embedders = {}
        keys = list(params.keys())
        for key in keys:
            embedder_params = params.pop(key)
            token_embedders[key] = TokenEmbedder.from_params(vocab, embedder_params)
        params.assert_empty(cls.__name__)
        return cls(token_embedders)
