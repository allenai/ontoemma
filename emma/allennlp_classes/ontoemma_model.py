from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder, FeedForward, TimeDistributed
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from emma.allennlp_classes.boolean_f1 import BooleanF1


@Model.register("ontoemmaNN")
class OntoEmmaNN(Model):

    def __init__(self, vocab: Vocabulary,
                 decision_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(OntoEmmaNN, self).__init__(vocab, regularizer)

        self.decision_feedforward = decision_feedforward
        self.sigmoid = torch.nn.Sigmoid()
        self.accuracy = BooleanF1()
        self.loss = torch.nn.BCELoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                lr_features: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Embed and encode entity name, aliases, definition, and contexts, run
        all through a feedforward network, aggregate the outputs and run through
        a decision layer.
        """
        # concatenate outputs
        aggregate_input = lr_features.squeeze(2).float()

        # run aggregate through a decision layer and sigmoid function
        decision_output = self.decision_feedforward(aggregate_input)

        sigmoid_output = self.sigmoid(decision_output)

        # build output dictionary
        output_dict = dict()
        output_dict["score"] = sigmoid_output
        predicted_label = sigmoid_output.round()
        output_dict["predicted_label"] = predicted_label

        if label is not None:
            # compute loss and accuracy
            loss = self.loss(sigmoid_output, label.float())
            self.accuracy(predicted_label, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, accuracy, f1 = self.accuracy.get_metric(reset)
        return {
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'f1_score': f1
        }

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'OntoEmmaNN':
        decision_feedforward = FeedForward.from_params(params.pop("decision_feedforward"))

        init_params = params.pop('initializer', None)
        reg_params = params.pop('regularizer', None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())
        regularizer = RegularizerApplicator.from_params(reg_params) if reg_params is not None else None

        return cls(vocab=vocab,
                   decision_feedforward=decision_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)