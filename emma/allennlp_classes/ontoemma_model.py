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
                 name_embedder: TextFieldEmbedder,
                 name_encoder: Seq2VecEncoder,
                 siamese_feedforward: FeedForward,
                 decision_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(OntoEmmaNN, self).__init__(vocab, regularizer)

        self.name_embedder = name_embedder
        self.name_encoder = name_encoder
        self.siamese_feedforward = siamese_feedforward
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

    @overrides
    def forward(self,  # type: ignore
                s_ent_name: Dict[str, torch.LongTensor],
                t_ent_name: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Embed and encode entity name, aliases, definition, and contexts, run
        all through a feedforward network, aggregate the outputs and run through
        a decision layer.
        """
        # embed and encode all aliases
        embedded_s_ent_aliases = self.distributed_name_embedder(s_ent_aliases)
        s_ent_aliases_mask = get_text_field_mask(s_ent_aliases)
        encoded_s_ent_aliases = TimeDistributed(self.name_rnn_encoder)(embedded_s_ent_aliases, s_ent_aliases_mask)

        embedded_t_ent_aliases = self.distributed_name_embedder(t_ent_aliases)
        t_ent_aliases_mask = get_text_field_mask(t_ent_aliases)
        encoded_t_ent_aliases = TimeDistributed(self.name_rnn_encoder)(embedded_t_ent_aliases, t_ent_aliases_mask)

        # average across non-zero entries
        best_alias_similarity, best_s_ent_alias, best_t_ent_alias = self._get_max_sim(
            encoded_s_ent_aliases, encoded_t_ent_aliases
        )

        # run both entity representations through feed forward network
        s_ent_output = self.siamese_feedforward(best_s_ent_alias)
        t_ent_output = self.siamese_feedforward(best_t_ent_alias)

        # concatenate outputs
        aggregate_input = torch.cat([
            s_ent_output,
            t_ent_output
        ], dim=-1)

        # run aggregate through a decision layer and sigmoid function
        decision_output = self.decision_feedforward(aggregate_input)
        sigmoid_output = self.sigmoid(decision_output)
        predicted_label = sigmoid_output.round()

        # build output dictionary
        output_dict = dict()
        output_dict["score"] = sigmoid_output
        output_dict["predicted_label"] = predicted_label
        output_dict["s_alias_encoding"] = s_ent_output
        output_dict["t_alias_encoding"] = t_ent_output

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
        name_embedder = TextFieldEmbedder.from_params(vocab, params.pop("name_embedder"))
        name_encoder = Seq2VecEncoder.from_params(params.pop("name_encoder"))
        siamese_feedforward = FeedForward.from_params(params.pop("siamese_feedforward"))
        decision_feedforward = FeedForward.from_params(params.pop("decision_feedforward"))

        init_params = params.pop('initializer', None)
        reg_params = params.pop('regularizer', None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())
        regularizer = RegularizerApplicator.from_params(reg_params) if reg_params is not None else None

        return cls(vocab=vocab,
                   name_embedder=name_embedder,
                   name_encoder=name_encoder,
                   siamese_feedforward=siamese_feedforward,
                   decision_feedforward=decision_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)