from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.service.predictors.predictor import Predictor


@Predictor.register('ontoemma-predictor')
class OntoEmmaPredictor(Predictor):
    """
    Wrapper for the :class:`~allennlp.models.ontoemma` model.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super(OntoEmmaPredictor, self).__init__(model, dataset_reader)

    @overrides
    def _json_to_instance(self, json: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``
        and returns JSON that looks like
        ``{"tags": [...], "class_probabilities": [[...], ..., [...]]}``
        """
        source_ent = json["source_ent"]
        target_ent = json["target_ent"]
        label = json["label"]
        return self._dataset_reader.text_to_instance(source_ent, target_ent, label)