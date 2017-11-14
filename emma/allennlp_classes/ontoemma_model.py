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
                 name_text_field_embedder: TextFieldEmbedder,
                 context_text_field_embedder: TextFieldEmbedder,
                 name_rnn_encoder: Seq2VecEncoder,
                 context_encoder: Seq2VecEncoder,
                 siamese_feedforward: FeedForward,
                 decision_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(OntoEmmaNN, self).__init__(vocab, regularizer)

        self.name_text_field_embedder = name_text_field_embedder
        self.distributed_name_embedder = BasicTextFieldEmbedder({
            k: TimeDistributed(v) for k, v in name_text_field_embedder._token_embedders.items()
        })
        self.context_text_field_embedder = context_text_field_embedder
        self.name_rnn_encoder = name_rnn_encoder
        self.context_encoder = context_encoder
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
                s_ent_name: Dict[str, torch.LongTensor],
                t_ent_name: Dict[str, torch.LongTensor],
                s_ent_aliases: Dict[str, torch.LongTensor],
                t_ent_aliases: Dict[str, torch.LongTensor],
                s_ent_def: Dict[str, torch.LongTensor],
                t_ent_def: Dict[str, torch.LongTensor],
                s_ent_context: Dict[str, torch.LongTensor],
                t_ent_context: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Embed and encode entity name, aliases, definition, and contexts, run
        all through a feedforward network, aggregate the outputs and run through
        a decision layer.
        """

        # embed and encode entity names
        embedded_s_ent_name = self.name_text_field_embedder(s_ent_name)
        s_ent_name_mask = get_text_field_mask(s_ent_name)
        encoded_s_ent_name = self.name_rnn_encoder(embedded_s_ent_name, s_ent_name_mask)

        embedded_t_ent_name = self.name_text_field_embedder(t_ent_name)
        t_ent_name_mask = get_text_field_mask(t_ent_name)
        encoded_t_ent_name = self.name_rnn_encoder(embedded_t_ent_name, t_ent_name_mask)

        name_similarity = torch.diag(encoded_s_ent_name.mm(encoded_t_ent_name.t()), 0)

        # embed and encode all aliases
        embedded_s_ent_aliases = self.distributed_name_embedder(s_ent_aliases)
        s_ent_aliases_mask = get_text_field_mask(s_ent_aliases)
        encoded_s_ent_aliases = TimeDistributed(self.name_rnn_encoder)(embedded_s_ent_aliases, s_ent_aliases_mask)

        embedded_t_ent_aliases = self.distributed_name_embedder(t_ent_aliases)
        t_ent_aliases_mask = get_text_field_mask(t_ent_aliases)
        encoded_t_ent_aliases = TimeDistributed(self.name_rnn_encoder)(embedded_t_ent_aliases, t_ent_aliases_mask)

        alias_max_similarity, best_s_aliases, best_t_aliases = self._get_max_sim(
            encoded_s_ent_aliases, encoded_t_ent_aliases
        )

        # embed and encode all definitions
        embedded_s_ent_def = self.context_text_field_embedder(s_ent_def)
        s_ent_def_mask = get_text_field_mask(s_ent_def)
        encoded_s_ent_def = self.context_encoder(embedded_s_ent_def, s_ent_def_mask)

        embedded_t_ent_def = self.context_text_field_embedder(t_ent_def)
        t_ent_def_mask = get_text_field_mask(t_ent_def)
        encoded_t_ent_def = self.context_encoder(embedded_t_ent_def, t_ent_def_mask)

        def_similarity = torch.diag(encoded_s_ent_def.mm(encoded_t_ent_def.t()), 0)

        # embed and encode all contexts
        embedded_s_ent_context = self.context_text_field_embedder(s_ent_context)
        s_ent_context_mask = get_text_field_mask(s_ent_context)
        encoded_s_ent_context = TimeDistributed(self.context_encoder)(embedded_s_ent_context, s_ent_context_mask)

        embedded_t_ent_context = self.context_text_field_embedder(t_ent_context)
        t_ent_context_mask = get_text_field_mask(t_ent_context)
        encoded_t_ent_context = TimeDistributed(self.context_encoder)(embedded_t_ent_context, t_ent_context_mask)

        context_max_similarity, best_s_context, best_t_context = self._get_max_sim(
            encoded_s_ent_context, encoded_t_ent_context
        )

        avg_s_context = self._get_avg(encoded_s_ent_context)
        avg_t_context = self._get_avg(encoded_t_ent_context)

        # embed and encode all parent relations
        embedded_s_ent_parents = self.distributed_name_embedder(s_ent_parents)
        s_ent_parents_mask = get_text_field_mask(s_ent_parents)
        encoded_s_ent_parents = TimeDistributed(self.name_rnn_encoder)(embedded_s_ent_parents, s_ent_parents_mask)

        embedded_t_ent_parents = self.distributed_name_embedder(t_ent_parents)
        t_ent_parents_mask = get_text_field_mask(t_ent_parents)
        encoded_t_ent_parents = TimeDistributed(self.name_rnn_encoder)(embedded_t_ent_parents, t_ent_parents_mask)

        avg_s_parents = self._get_avg(encoded_s_ent_parents)
        avg_t_parents = self._get_avg(encoded_t_ent_parents)

        # embed and encode all child relations
        embedded_s_ent_children = self.distributed_name_embedder(s_ent_children)
        s_ent_children_mask = get_text_field_mask(s_ent_children)
        encoded_s_ent_children = TimeDistributed(self.name_rnn_encoder)(embedded_s_ent_children, s_ent_children_mask)

        embedded_t_ent_children = self.distributed_name_embedder(t_ent_children)
        t_ent_children_mask = get_text_field_mask(t_ent_children)
        encoded_t_ent_children = TimeDistributed(self.name_rnn_encoder)(embedded_t_ent_children, t_ent_children_mask)

        avg_s_children = self._get_avg(encoded_s_ent_children)
        avg_t_children = self._get_avg(encoded_t_ent_children)

        # input into feed forward network (placeholder for concatenating other features)
        s_ent_input = torch.cat(
            [encoded_s_ent_name,
             best_s_aliases,
             encoded_s_ent_def,
             avg_s_context,
             avg_s_parents,
             avg_s_children
             ],
            dim=-1)
        t_ent_input = torch.cat(
            [encoded_t_ent_name,
             best_t_aliases,
             encoded_t_ent_def,
             avg_t_context,
             avg_t_parents,
             avg_t_children
             ],
            dim=-1)

        # run both entity representations through feed forward network
        s_ent_output = self.siamese_feedforward(s_ent_input)
        t_ent_output = self.siamese_feedforward(t_ent_input)

        # aggregate similarity metrics
        aggregate_similarity = torch.stack(
            [name_similarity,
             alias_max_similarity,
             def_similarity,
             context_max_similarity
             ], dim=-1
        )

        # concatenate outputs
        aggregate_input = torch.cat([aggregate_similarity, s_ent_output, t_ent_output], dim=-1)

        # run aggregate through a decision layer and sigmoid function
        decision_output = self.decision_feedforward(aggregate_input)

        sigmoid_output = self.sigmoid(decision_output)

        # round sigmoid output to get prediction label
        predicted_label = sigmoid_output.round()

        # build output dictionary
        output_dict = {"predicted_label": predicted_label}

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
        name_text_field_embedder = TextFieldEmbedder.from_params(vocab, params.pop("name_text_field_embedder"))
        context_text_field_embedder = TextFieldEmbedder.from_params(vocab, params.pop("context_text_field_embedder"))
        name_rnn_encoder = Seq2VecEncoder.from_params(params.pop("name_rnn_encoder"))
        context_encoder = Seq2VecEncoder.from_params(params.pop("context_encoder"))
        siamese_feedforward = FeedForward.from_params(params.pop("siamese_feedforward"))
        decision_feedforward = FeedForward.from_params(params.pop("decision_feedforward"))

        init_params = params.pop('initializer', None)
        reg_params = params.pop('regularizer', None)
        initializer = (InitializerApplicator.from_params(init_params)
                       if init_params is not None
                       else InitializerApplicator())
        regularizer = RegularizerApplicator.from_params(reg_params) if reg_params is not None else None

        return cls(vocab=vocab,
                   name_text_field_embedder=name_text_field_embedder,
                   context_text_field_embedder=context_text_field_embedder,
                   name_rnn_encoder=name_rnn_encoder,
                   context_encoder=context_encoder,
                   siamese_feedforward=siamese_feedforward,
                   decision_feedforward=decision_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)