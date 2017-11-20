from overrides import overrides
from typing import Dict, List
import random
import torch
import numpy

from allennlp.common.util import JsonDict, sanitize
from allennlp.nn.util import arrays_to_variables
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor
from allennlp.data.fields import Field, TextField, ListField
from allennlp.nn.util import get_text_field_mask
from allennlp.modules import TimeDistributed
from allennlp.data.dataset import Dataset

@Predictor.register('ontoemma-predictor')
class OntoEmmaPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.ontoemma` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super(OntoEmmaPredictor, self).__init__(model, dataset_reader)

    @overrides
    def predict_json(self, inputs: JsonDict, cuda_device: int = -1) -> JsonDict:
        instance = self._json_to_instance(inputs)
        instance.index_fields(self._model.vocab)
        model_input = arrays_to_variables(instance.as_array_dict(),
                                          add_batch_dimension=True,
                                          cuda_device=cuda_device,
                                          for_training=False)
        outputs = self.decode(self._get_encoding_for_instance(**model_input))

        for name, output in list(outputs.items()):
            # We are predicting on a single instance and we added a batch
            # dimension, so here we remove it.
            output = output[0]
            if isinstance(output, torch.autograd.Variable):
                output = output.data.cpu().numpy()
            outputs[name] = output
        return sanitize(outputs)

    @overrides
    def _json_to_instance(self, json: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``
        and returns JSON that looks like
        ``{"tags": [...], "class_probabilities": [[...], ..., [...]]}``
        """

        # sample n from list l, keeping only entries with len less than max_len
        # if n is greater than the length of l, just return l
        def sample_n(l, n, max_len):
            l = [i for i in l if len(i) <= max_len]
            if not l:
                return ['00000']
            if len(l) <= n:
                return l
            return random.sample(l, n)

        fields: Dict[str, Field] = {}
        # name field
        name_tokens = self._dataset_reader._tokenizer.tokenize('00000 00000 00000 00000 ' + json['canonical_name'])
        fields['name'] = TextField(name_tokens, self._dataset_reader._name_token_indexers)

        # alias field
        aliases = sample_n(json['aliases'], 16, 128)
        fields['aliases'] = ListField(
            [TextField(self._dataset_reader._tokenizer.tokenize('00000 00000 00000 00000 ' + a),
                       self._dataset_reader._name_token_indexers)
             for a in aliases]
        )

        # definition field
        fields['definition'] = TextField(
            self._dataset_reader._tokenizer.tokenize(json['definition']),
            self._dataset_reader._token_only_indexer
        ) if json['definition'] \
            else TextField(self._dataset_reader._tokenizer.tokenize('00000'),
                           self._dataset_reader._token_only_indexer)

        # context field
        contexts = sample_n(json['other_contexts'], 16, 256)

        fields['contexts'] = ListField(
            [TextField(self._dataset_reader._tokenizer.tokenize(c),
                       self._dataset_reader._token_only_indexer)
             for c in contexts]
        )

        return Instance(fields)

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict], cuda_device: int = -1) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        dataset = Dataset(instances)
        dataset.index_instances(self._model.vocab)

        model_input = arrays_to_variables(dataset.as_array_dict(),
                                          cuda_device=cuda_device,
                                          for_training=False)

        outputs = self._model.decode(self._get_encoding_for_instance(**model_input))

        instance_separated_output: List[Dict[str, numpy.ndarray]] = [{} for _ in dataset.instances]
        for name, output in list(outputs.items()):
            if isinstance(output, torch.autograd.Variable):
                output = output.data.cpu().numpy()
            outputs[name] = output
            for instance_output, batch_element in zip(instance_separated_output, output):
                instance_output[name] = batch_element
        return sanitize(instance_separated_output)

    def _get_encoding_for_instance(self, name, aliases, definition, contexts):

        embedded_name = self._model.name_text_field_embedder(name)
        name_mask = get_text_field_mask(name)
        encoded_name = self._model.name_cnn_encoder(embedded_name, name_mask)

        embedded_aliases = self._model.distributed_name_embedder(aliases)
        aliases_mask = get_text_field_mask(aliases)
        encoded_aliases = TimeDistributed(self._model.name_cnn_encoder)(embedded_aliases, aliases_mask)

        aliases_mask = torch.sum(encoded_aliases, 2) != 0.0
        averaged_aliases = self._model.name_boe_encoder(encoded_aliases, aliases_mask)

        embedded_def = self._model.context_text_field_embedder(definition)
        def_mask = get_text_field_mask(definition)
        encoded_def = self._model.context_encoder(embedded_def, def_mask)

        embedded_context = self._model.context_text_field_embedder(contexts)
        context_mask = get_text_field_mask(contexts)
        encoded_context = TimeDistributed(self._model.context_encoder)(embedded_context, context_mask)

        context_mask = torch.sum(encoded_context, 2) != 0.0
        averaged_context = self._model.context_encoder(encoded_context, context_mask)

        ent_input = torch.cat(
            [encoded_name,
             averaged_aliases,
             encoded_def,
             averaged_context
             ],
            dim=-1)

        ent_output = self._model.siamese_feedforward(ent_input)

        output_dict = dict()
        output_dict['ent_rep'] = ent_output
        return output_dict