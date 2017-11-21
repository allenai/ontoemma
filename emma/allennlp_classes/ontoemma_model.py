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

    @staticmethod
    def _get_max_sim(s_stack, t_stack):
        """
        Get max similarity of each pair of corresponding entries from two ListFields
        :param s_stack:
        :param t_stack:
        :return: max similarities, best field in s (max sim), best field in t (max sim)
        """
        max_vals = []
        best_s = []
        best_t = []

        for s_entry, t_entry in zip(s_stack, t_stack):
            s_maxvals, sidx = torch.max(s_entry.mm(t_entry.t()), 0)
            if s_maxvals.dim() == 1:
                s_max = torch.max(s_maxvals)
                tidx = 0
            else:
                s_max, tidx = torch.max(s_maxvals, 1)
            sidx = sidx.squeeze()[tidx]
            max_vals.append(s_max)
            best_s.append(s_entry[sidx].squeeze())
            best_t.append(t_entry[tidx].squeeze())

        return torch.stack(max_vals, 0).squeeze(-1), \
               torch.stack(best_s, 0), torch.stack(best_t, 0)

    @staticmethod
    def _get_avg(stack):
        """
        Compute average over non-zero entries
        :param stack:
        :return:
        """
        avg_vec = []
        for entry in stack:
            sums = torch.sum(entry, 1)
            nonzero = torch.nonzero(sums.data).size()
            if len(nonzero) > 0:
                avg_vec.append(torch.sum(entry, 0) / nonzero[0])
            else:
                avg_vec.append(entry[0])

        return torch.stack(avg_vec, 0)

    @overrides
    def forward(self,  # type: ignore
                has_same_canonical_name: Dict[str, torch.LongTensor],
                has_same_stemmed_name: Dict[str, torch.LongTensor],
                has_same_lemmatized_name: Dict[str, torch.LongTensor],
                has_same_char_tokens: Dict[str, torch.LongTensor],
                has_alias_in_common: Dict[str, torch.LongTensor],
                name_token_jaccard: Dict[str, torch.LongTensor],
                inverse_name_edit_distance: Dict[str, torch.LongTensor],
                stemmed_token_jaccard: Dict[str, torch.LongTensor],
                inverse_stemmed_edit_distance: Dict[str, torch.LongTensor],
                lemmatized_token_jaccard: Dict[str, torch.LongTensor],
                inverse_lemmatized_edit_distance: Dict[str, torch.LongTensor],
                char_token_jaccard: Dict[str, torch.LongTensor],
                inverse_char_token_edit_distance: Dict[str, torch.LongTensor],
                max_alias_token_jaccard: Dict[str, torch.LongTensor],
                inverse_min_alias_edit_distance: Dict[str, torch.LongTensor],
                percent_parents_in_common: Dict[str, torch.LongTensor],
                percent_children_in_common: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Embed and encode entity name, aliases, definition, and contexts, run
        all through a feedforward network, aggregate the outputs and run through
        a decision layer.
        """

        # aggregate similarity metrics
        aggregate_similarity = torch.stack(
            [has_same_canonical_name,
             has_same_stemmed_name,
             has_same_lemmatized_name,
             has_same_char_tokens,
             has_alias_in_common,
             name_token_jaccard,
             inverse_name_edit_distance,
             stemmed_token_jaccard,
             inverse_stemmed_edit_distance,
             lemmatized_token_jaccard,
             inverse_lemmatized_edit_distance,
             char_token_jaccard,
             inverse_char_token_edit_distance,
             max_alias_token_jaccard,
             inverse_min_alias_edit_distance,
             percent_parents_in_common,
             percent_children_in_common
             ], dim=-1
        )

        # squeeze
        aggregate_input = aggregate_similarity.squeeze(1).float()

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