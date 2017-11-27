from typing import Dict
import logging

from overrides import overrides
import numpy

from allennlp.data.fields.field import Field

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FloatField(Field[numpy.ndarray]):
    """
    A ``FloatField`` contains a float value
    """
    def __init__(self,
                 value: float):
        self.value = value

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:  # pylint: disable=no-self-use
        return {}

    @overrides
    def as_array(self, padding_lengths: Dict[str, int]) -> numpy.ndarray:  # pylint: disable=unused-argument
        return numpy.asarray([self.value])

    @overrides
    def empty_field(self):
        return FloatField(0.0)