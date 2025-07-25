
import random as _rd
from collections import abc as _abc
import itertools as _it
import pseudowords as _pw

def get_data(
        word_bank : _abc.Sequence[str], 
        num_samples : int, 
        real_word_predicate : _abc.Callable[[str], bool], 
        pseudoword_predicate : _abc.Callable[[str], bool],
        pseudoword_fraction : float
        , /
    ) -> _abc.Sequence[tuple[str, bool]]:

    """
    Args:
        `word_bank`: contains real words.
        `num_samples`: positive integer.
        `real_word_predicate`: determines whether a real word is suitable to be used as training/testing data.
        `pseudoword`: determines whether a pseudoword is suitable to be used as training/testing data.
        `pseudoword_fraction`: the approximate fraction of the data that should be pseudowords.

    Returns:
        A sequence of `(word, realness)` pairs where each `word` in the sequence appears exactly once.
        
        For each word, either:
            `word` is in `word_bank`, `word_predicate(word)` is `True and `realness` is `True`
        or:
            `word` is a pseudoword, `pseudoword_predicate(word)` is True, and `realness` is `False`.
        `
        Up to rounding, the fraction of pseudowords in this sequence is `pseudoword_fraction`

    """

    real_word_fraction = 1 - pseudoword_fraction
    num_real_words = round(num_samples * real_word_fraction)
    num_pseudowords = num_samples - num_real_words

    bounded_real_words = tuple(word for word in word_bank if real_word_predicate(word))
    real_words = tuple(_rd.sample(bounded_real_words, num_real_words))

    pseudoword_generator = _pw.words_to_pseudoword_generator(word_bank)
    bounded_pseudoword_generator = (word for word in pseudoword_generator if pseudoword_predicate(word))
    del pseudoword_generator

    pseudowords : set[str] = set()

    while len(pseudowords) < num_pseudowords:
        word = next(bounded_pseudoword_generator)
        pseudowords.add(word)

    del bounded_pseudoword_generator

    pairs = list(_it.chain(zip(real_words, _it.repeat(True)), zip(pseudowords, _it.repeat(False))))
    _rd.shuffle(pairs)

    return pairs
