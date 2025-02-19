import itertools
import logging

logger = logging.getLogger(__name__)


def generator_length(gen):
    """
    compute the no of elems inside a generator
    :param gen:
    :return:
    """
    gen1, gen2 = itertools.tee(gen)
    length = sum(1 for _ in gen1)  # Consume the duplicate
    return length, gen2  # Return the length and the unconsumed generator
