# type: ignore

# Permitted importers: train_and_evaluate.py.
import tensorflow as _tf
from collections import abc as _abc
import numpy as _np
import typing as _tp
import pathlib as _pt

class KerasModelWrapper:

    """ An externally type-safe wrapper for a Keras model tuned for the purpose of the word-pseudoword classification task."""

    _model : _tp.Any

    def __init__(self, /) -> object:
        pass

    def train(self, x_training_seq : _abc.Sequence[_abc.Sequence[int]], y_training_seq : _abc.Sequence[float], /) -> object:
        x_training = _np.array(x_training_seq)
        x_training_onehot = _tf.one_hot(x_training, 27).numpy()
        y_training = _np.array(y_training_seq).reshape(-1, 1)
        max_word_length, vocab_size = x_training_onehot.shape[1], x_training_onehot.shape[2]

        self._model = _tf.keras.Sequential([
            _tf.keras.layers.Input(shape=(max_word_length, vocab_size)), 
            _tf.keras.layers.Conv1D(200, kernel_size=3, activation='relu', padding='same'),
            _tf.keras.layers.Conv1D(200, kernel_size=3, activation='relu', padding='same'),
            _tf.keras.layers.GlobalMaxPooling1D(),
            _tf.keras.layers.Dense(50, activation='relu'),
            _tf.keras.layers.Dense(1, activation='sigmoid')

        ])
        
        self._model.compile(optimizer=_tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['binary_crossentropy', 'binary_accuracy'])
        self._model.fit(x_training_onehot, y_training, epochs=15, batch_size=128, validation_split=0.1)

    def evaluate(self, x_testing_seq : _abc.Sequence[_abc.Sequence[int]], y_testing_seq : _abc.Sequence[float], /) -> tuple[float, float, float]:
        x_testing = _np.array(x_testing_seq)
        x_testing_onehot = _tf.one_hot(x_testing, 27).numpy() 
        y_testing = _np.array(y_testing_seq).reshape(-1, 1) 
        
        return self._model.evaluate(x_testing_onehot, y_testing)

    def predict(self, xs : _abc.Sequence[_abc.Sequence[int]], /) -> _abc.Sequence[float]:
        x_onehot = _tf.one_hot(_np.array(xs), 27).numpy()
        return self._model.predict(x_onehot).flatten()

    def save(self, file_path : str | _pt.Path, /) -> object:
        self._model.save(file_path)
    
    def load(self, file_path : str | _pt.Path, /) -> object:
        self._model = _tf.keras.models.load_model(file_path)

