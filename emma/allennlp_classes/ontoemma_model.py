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
                 name_cnn_encoder: Seq2VecEncoder,
                 name_boe_encoder: Seq2VecEncoder,
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
        self.name_cnn_encoder = name_cnn_encoder
        self.name_boe_encoder = name_boe_encoder
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
        encoded_s_ent_name = self.name_cnn_encoder(embedded_s_ent_name, s_ent_name_mask)

        embedded_t_ent_name = self.name_text_field_embedder(t_ent_name)
        t_ent_name_mask = get_text_field_mask(t_ent_name)
        encoded_t_ent_name = self.name_cnn_encoder(embedded_t_ent_name, t_ent_name_mask)

        name_similarity = torch.diag(encoded_s_ent_name.mm(encoded_t_ent_name.t()), 0)

        # embed and encode all aliases
        embedded_s_ent_aliases = self.distributed_name_embedder(s_ent_aliases)
        s_ent_aliases_mask = get_text_field_mask(s_ent_aliases)
        encoded_s_ent_aliases = TimeDistributed(self.name_cnn_encoder)(embedded_s_ent_aliases, s_ent_aliases_mask)

        s_ent_aliases_mask = torch.sum(encoded_s_ent_aliases, 2) != 0.0
        averaged_s_ent_aliases = self.name_boe_encoder(encoded_s_ent_aliases, s_ent_aliases_mask)

        embedded_t_ent_aliases = self.distributed_name_embedder(t_ent_aliases)
        t_ent_aliases_mask = get_text_field_mask(t_ent_aliases)
        encoded_t_ent_aliases = TimeDistributed(self.name_cnn_encoder)(embedded_t_ent_aliases, t_ent_aliases_mask)

        t_ent_aliases_mask = torch.sum(encoded_t_ent_aliases, 2) != 0.0
        averaged_t_ent_aliases = self.name_boe_encoder(encoded_t_ent_aliases, t_ent_aliases_mask)

        alias_similarity = torch.diag(averaged_s_ent_aliases.mm(averaged_t_ent_aliases.t()), 0)

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

        s_ent_context_mask = torch.sum(encoded_s_ent_context, 2) != 0.0
        averaged_s_ent_context = self.context_encoder(encoded_s_ent_context, s_ent_context_mask)

        embedded_t_ent_context = self.context_text_field_embedder(t_ent_context)
        t_ent_context_mask = get_text_field_mask(t_ent_context)
        encoded_t_ent_context = TimeDistributed(self.context_encoder)(embedded_t_ent_context, t_ent_context_mask)

        t_ent_context_mask = torch.sum(encoded_t_ent_context, 2) != 0.0
        averaged_t_ent_context = self.context_encoder(encoded_t_ent_context, t_ent_context_mask)

        context_similarity = torch.diag(averaged_s_ent_context.mm(averaged_t_ent_context.t()), 0)

        # input into feed forward network (placeholder for concatenating other features)
        s_ent_input = torch.cat(
            [encoded_s_ent_name,
             averaged_s_ent_aliases,
             encoded_s_ent_def,
             averaged_s_ent_context
             ],
            dim=-1)
        t_ent_input = torch.cat(
            [encoded_t_ent_name,
             averaged_t_ent_aliases,
             encoded_t_ent_def,
             averaged_t_ent_context
             ],
            dim=-1)

        # run both entity representations through feed forward network
        s_ent_output = self.siamese_feedforward(s_ent_input)
        t_ent_output = self.siamese_feedforward(t_ent_input)

        # aggregate similarity metrics
        aggregate_similarity = torch.stack(
            [name_similarity,
             alias_similarity,
             def_similarity,
             context_similarity
             ], dim=-1
        )

        # concatenate outputs
        aggregate_input = torch.cat([aggregate_similarity, s_ent_output, t_ent_output], dim=-1)

        # run aggregate through a decision layer and sigmoid function
        decision_output = self.decision_feedforward(aggregate_input)

        sigmoid_output = self.sigmoid(decision_output)

        # build output dictionary
        output_dict = dict()
        output_dict["s_ent_rep"] = s_ent_output
        output_dict["t_ent_rep"] = t_ent_output
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
        name_text_field_embedder = TextFieldEmbedder.from_params(vocab, params.pop("name_text_field_embedder"))
        context_text_field_embedder = TextFieldEmbedder.from_params(vocab, params.pop("context_text_field_embedder"))
        name_cnn_encoder = Seq2VecEncoder.from_params(params.pop("name_cnn_encoder"))
        name_boe_encoder = Seq2VecEncoder.from_params(params.pop("name_boe_encoder"))
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
                   name_cnn_encoder=name_cnn_encoder,
                   name_boe_encoder=name_boe_encoder,
                   context_encoder=context_encoder,
                   siamese_feedforward=siamese_feedforward,
                   decision_feedforward=decision_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)