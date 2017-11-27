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
        self.distributed_context_embedder = BasicTextFieldEmbedder({
            k: TimeDistributed(v) for k, v in context_text_field_embedder._token_embedders.items()
        })
        self.name_rnn_encoder = name_rnn_encoder
        self.context_encoder = context_encoder
        self.siamese_feedforward = siamese_feedforward
        self.decision_feedforward = decision_feedforward
        self.sigmoid = torch.nn.Sigmoid()
        self.accuracy = BooleanF1()
        self.loss = torch.nn.BCELoss()

        initializer(self)

    @staticmethod
    def _average_nonzero(t_stack):
        output_rows = []
        for row in t_stack:
            if row.sum().data[0] == 0.0:
                output_rows.append(row[0])
            else:
                output_rows.append(row.sum(0) / ((row.sum(1) != 0.0).sum().data[0]))

        return torch.stack(output_rows)


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

        # embed and encode all aliases
        embedded_s_ent_aliases = self.distributed_name_embedder(s_ent_aliases)
        s_ent_aliases_mask = get_text_field_mask(s_ent_aliases)
        encoded_s_ent_aliases = TimeDistributed(self.name_rnn_encoder)(embedded_s_ent_aliases, s_ent_aliases_mask)

        embedded_t_ent_aliases = self.distributed_name_embedder(t_ent_aliases)
        t_ent_aliases_mask = get_text_field_mask(t_ent_aliases)
        encoded_t_ent_aliases = TimeDistributed(self.name_rnn_encoder)(embedded_t_ent_aliases, t_ent_aliases_mask)

        # average across non-zero entries
        average_encoded_s_ent_aliases = self._average_nonzero(encoded_s_ent_aliases)
        average_encoded_t_ent_aliases = self._average_nonzero(encoded_t_ent_aliases)

        # embed and encode all definitions
        embedded_s_ent_def = self.context_text_field_embedder(s_ent_def)
        s_ent_def_mask = get_text_field_mask(s_ent_def)
        encoded_s_ent_def = self.context_encoder(embedded_s_ent_def, s_ent_def_mask)

        embedded_t_ent_def = self.context_text_field_embedder(t_ent_def)
        t_ent_def_mask = get_text_field_mask(t_ent_def)
        encoded_t_ent_def = self.context_encoder(embedded_t_ent_def, t_ent_def_mask)

        # embed and encode all contexts
        embedded_s_ent_context = self.distributed_context_embedder(s_ent_context)
        s_ent_context_mask = get_text_field_mask(s_ent_context)
        encoded_s_ent_context = TimeDistributed(self.context_encoder)(embedded_s_ent_context, s_ent_context_mask)

        embedded_t_ent_context = self.distributed_context_embedder(t_ent_context)
        t_ent_context_mask = get_text_field_mask(t_ent_context)
        encoded_t_ent_context = TimeDistributed(self.context_encoder)(embedded_t_ent_context, t_ent_context_mask)

        # average contexts
        average_encoded_s_ent_context = self._average_nonzero(encoded_s_ent_context)
        average_encoded_t_ent_context = self._average_nonzero(encoded_t_ent_context)

        # input into feed forward network (placeholder for concatenating other features)
        s_ent_input = torch.cat(
            [encoded_s_ent_name,
             average_encoded_s_ent_aliases,
             encoded_s_ent_def,
             average_encoded_s_ent_context
             ],
            dim=-1)
        t_ent_input = torch.cat(
            [encoded_t_ent_name,
             average_encoded_t_ent_aliases,
             encoded_t_ent_def,
             average_encoded_t_ent_context
             ],
            dim=-1)

        # run both entity representations through feed forward network
        s_ent_output = self.siamese_feedforward(s_ent_input)
        t_ent_output = self.siamese_feedforward(t_ent_input)

        # concatenate outputs
        aggregate_input = torch.cat([s_ent_output, t_ent_output], dim=-1)

        # run aggregate through a decision layer and sigmoid function
        decision_output = self.decision_feedforward(aggregate_input)

        sigmoid_output = self.sigmoid(decision_output)

        # round sigmoid output to get prediction label
        predicted_label = sigmoid_output.round()

        # build output dictionary
        output_dict = {"predicted_label": predicted_label}
        output_dict["score"] = sigmoid_output

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