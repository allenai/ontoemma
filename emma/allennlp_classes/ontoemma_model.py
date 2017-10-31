from typing import Dict, Optional

from overrides import overrides
import torch
import itertools

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2VecEncoder, TextFieldEmbedder, FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask

from emma.allennlp_classes.boolean_f1 import BooleanF1


@Model.register("ontoemmaNN")
class OntoEmmaNN(Model):

    def __init__(self, vocab: Vocabulary,
                 name_text_field_embedder: TextFieldEmbedder,
                 alias_text_field_embedder: TextFieldEmbedder,
                 definition_text_field_embedder: TextFieldEmbedder,
                 context_text_field_embedder: TextFieldEmbedder,
                 name_rnn_encoder: Seq2VecEncoder,
                 definition_encoder: Seq2VecEncoder,
                 context_encoder: Seq2VecEncoder,
                 siamese_feedforward: FeedForward,
                 decision_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(OntoEmmaNN, self).__init__(vocab, regularizer)

        self.name_text_field_embedder = name_text_field_embedder
        self.alias_text_field_embedder = alias_text_field_embedder
        self.definition_text_field_embedder = definition_text_field_embedder
        self.context_text_field_embedder = context_text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.name_rnn_encoder = name_rnn_encoder
        self.definition_encoder = definition_encoder
        self.context_encoder = context_encoder
        self.siamese_feedforward = siamese_feedforward
        self.decision_feedforward = decision_feedforward
        self.sigmoid = torch.nn.Sigmoid()

        if name_text_field_embedder.get_output_dim() != name_rnn_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the name_text_field_embedder must match the "
                                     "input dimension of the name_rnn_encoder. Found {} and {}, "
                                     "respectively.".format(name_text_field_embedder.get_output_dim(),
                                                            name_rnn_encoder.get_input_dim()))

        self.accuracy = BooleanF1()
        self.loss = torch.nn.BCELoss()

        initializer(self)

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
        embedded_s_ent_aliases = self.alias_text_field_embedder(s_ent_aliases)
        s_ent_aliases_mask = get_text_field_mask(s_ent_aliases)
        encoded_s_ent_aliases = []
        for row_embedding, row_mask in zip(embedded_s_ent_aliases, s_ent_aliases_mask):
            encoded_alias = self.name_rnn_encoder(row_embedding, row_mask)
            encoded_s_ent_aliases.append(encoded_alias)

        embedded_t_ent_aliases = self.alias_text_field_embedder(t_ent_aliases)
        t_ent_aliases_mask = get_text_field_mask(t_ent_aliases)
        encoded_t_ent_aliases = []
        for row_embedding, row_mask in zip(embedded_t_ent_aliases, t_ent_aliases_mask):
            encoded_alias = self.name_rnn_encoder(row_embedding, row_mask)
            encoded_t_ent_aliases.append(encoded_alias)

        # average aliases
        average_encoded_s_ent_aliases = []
        average_encoded_t_ent_aliases = []
        for s_alias, t_alias in zip(encoded_s_ent_aliases, encoded_t_ent_aliases):
            keep_s_alias = torch.stack(
                [alias for alias in s_alias if alias.sum().data[0] != 0.0])
            keep_t_alias = torch.stack(
                [alias for alias in t_alias if alias.sum().data[0] != 0.0])
            average_encoded_s_ent_aliases.append(keep_s_alias.mean(0))
            average_encoded_t_ent_aliases.append(keep_t_alias.mean(0))

        average_encoded_s_ent_aliases = torch.stack(average_encoded_s_ent_aliases)
        average_encoded_t_ent_aliases = torch.stack(average_encoded_t_ent_aliases)

        # embed and encode all definitions
        embedded_s_ent_def = self.definition_text_field_embedder(s_ent_def)
        s_ent_def_mask = get_text_field_mask(s_ent_def)
        encoded_s_ent_def = self.definition_encoder(embedded_s_ent_def, s_ent_def_mask)

        embedded_t_ent_def = self.definition_text_field_embedder(t_ent_def)
        t_ent_def_mask = get_text_field_mask(t_ent_def)
        encoded_t_ent_def = self.definition_encoder(embedded_t_ent_def, t_ent_def_mask)

        # embed and encode all contexts
        embedded_s_ent_context = self.context_text_field_embedder(s_ent_context)
        s_ent_context_mask = get_text_field_mask(s_ent_context)
        encoded_s_ent_context = []
        for row_embedding, row_mask in zip(embedded_s_ent_context, s_ent_context_mask):
            encoded_context = self.context_encoder(row_embedding, row_mask)
            encoded_s_ent_context.append(encoded_context)

        embedded_t_ent_context = self.context_text_field_embedder(t_ent_context)
        t_ent_context_mask = get_text_field_mask(t_ent_context)
        encoded_t_ent_context = []
        for row_embedding, row_mask in zip(embedded_t_ent_context, t_ent_context_mask):
            encoded_context = self.context_encoder(row_embedding, row_mask)
            encoded_t_ent_context.append(encoded_context)

        # average contexts
        average_encoded_s_ent_context = []
        average_encoded_t_ent_context = []
        for s_context, t_context in zip(encoded_s_ent_context, encoded_t_ent_context):
            keep_s_context = torch.stack(
                [context for context in s_context if context.sum().data[0] != 0.0])
            keep_t_context = torch.stack(
                [context for context in t_context if context.sum().data[0] != 0.0])
            average_encoded_s_ent_context.append(keep_s_context.mean(0))
            average_encoded_t_ent_context.append(keep_t_context.mean(0))

        average_encoded_s_ent_context = torch.stack(average_encoded_s_ent_context)
        average_encoded_t_ent_context = torch.stack(average_encoded_t_ent_context)

        # input into feed forward network (placeholder for concatenating other features)
        s_ent_input = torch.cat(
            [encoded_s_ent_name,
             average_encoded_s_ent_aliases,
             encoded_s_ent_def,
             average_encoded_s_ent_context],
            dim=-1)
        t_ent_input = torch.cat(
            [encoded_t_ent_name,
             average_encoded_t_ent_aliases,
             encoded_t_ent_def,
             average_encoded_t_ent_context],
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
        alias_text_field_embedder = TextFieldEmbedder.from_params(vocab, params.pop("alias_text_field_embedder"))
        definition_text_field_embedder = TextFieldEmbedder.from_params(vocab,
                                                                       params.pop("definition_text_field_embedder"))
        context_text_field_embedder = TextFieldEmbedder.from_params(vocab, params.pop("context_text_field_embedder"))
        name_rnn_encoder = Seq2VecEncoder.from_params(params.pop("name_rnn_encoder"))
        definition_encoder = Seq2VecEncoder.from_params(params.pop("definition_encoder"))
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
                   alias_text_field_embedder=alias_text_field_embedder,
                   definition_text_field_embedder=definition_text_field_embedder,
                   context_text_field_embedder=context_text_field_embedder,
                   name_rnn_encoder=name_rnn_encoder,
                   definition_encoder=definition_encoder,
                   context_encoder=context_encoder,
                   siamese_feedforward=siamese_feedforward,
                   decision_feedforward=decision_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)