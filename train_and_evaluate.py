
from collections import abc as _abc
import itertools as _it
import pathlib as _pt

import data as _dt 
import _keras_model_wrapper as _kmw

def _reformat_word(word : str, max_length : int, /) -> _abc.Sequence[int]:
    return tuple(ord(c) - 96 for c in word) + tuple(_it.repeat(0, (max_length - len(word))))

def _reformat_data(data : _abc.Sequence[tuple[str, bool]], max_word_length : int, /) -> tuple[_abc.Sequence[_abc.Sequence[int]], _abc.Sequence[float]]:
    xs = [_reformat_word(word, max_word_length) for word, _ in data]
    ys = [float(realness) for _, realness in data]
    return xs, ys

def _filter_alpha(s : str, /) -> str:
    return ''.join(c for c in s if c.isalpha())

def train_and_evaluate(write_file_path : str | _pt.Path, /) -> object:

    ## configurables ##
    word_file_path = '194000.txt'
    num_samples = 350_000
    min_word_length = 1
    max_word_length = 12
    pseudoword_fraction = 0.5
    testing_fraction = 0.1
    ## end of configurables ##

    num_training_samples = round(num_samples * (1 - testing_fraction))
    

    with open(word_file_path) as file:
        file_words = set(_filter_alpha(line[:-1]) for line in file.readlines())


    def real_word_predicate(word : str) -> bool:
        return  min_word_length <= len(word) <= max_word_length

    def pseudoword_predicate(word : str) -> bool:
        return real_word_predicate(word)

    xs, ys = _reformat_data(
        _dt.get_data(
            tuple(file_words), 
            num_samples, 
            real_word_predicate, 
            pseudoword_predicate,
            pseudoword_fraction), 
        max_word_length)

    x_training, y_training = xs[:num_training_samples], ys[:num_training_samples]
    x_testing, y_testing = xs[num_training_samples:], ys[num_training_samples:]

    model = _kmw.KerasModelWrapper()
    
    model.train(x_training, y_training)

    _, _, testing_accuracy = model.evaluate(x_testing, y_testing)

    print(f"Testing accuracy: {testing_accuracy}")
    
    model.save(write_file_path)

class Model:

    """
    A wrapper tuned for the purpose of the word-pseudoword classification task.  This wrapper is meant to provide a high level API written in high-level domain terms.
    """

    def __init__(self, max_word_length : int, /) -> None:
        self._model = _kmw.KerasModelWrapper()
        self._max_word_length = max_word_length

    def train(self, training_data : _abc.Sequence[tuple[str, bool]], /) -> object:
        xs, ys = _reformat_data(training_data, self._max_word_length)
        self._model.train(xs, ys)

    def evaluate(self, testing_data : _abc.Sequence[tuple[str, bool]], /) -> tuple[float, float, float]:
        xs, ys = _reformat_data(testing_data, self._max_word_length)
        return self._model.evaluate(xs, ys)

    def predict(self, words : _abc.Sequence[str], /) -> _abc.Sequence[float]:
        xs = [_reformat_word(word, self._max_word_length) for word in words]
        return self._model.predict(xs)
    
    def save(self, file_path : str | _pt.Path, /) -> object:
        self._model.save(file_path)
    
    def load(self, file_path : str | _pt.Path, /) -> object:
        self._model.load(file_path)

if __name__ == '__main__':
    train_and_evaluate('word_pseudoword_model.keras')

