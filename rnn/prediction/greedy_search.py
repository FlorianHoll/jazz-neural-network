"""
Implementation of Greedy Search for the Jazz Recurrent Neural Network.

Greedy Search is the easiest way of predicting from a Neural network; the most
probable value at each step is simply taken to be the next value.
"""
import keras
import numpy as np


class GreedySearch:
    """Greedy Search.

    :param model: The model to predict from with weights already loaded.
    :param seq_length: The length of one sequence passed to the neural net.
    """

    def __init__(self, model: keras.Model, seq_length: int = 8) -> None:
        """Initialize the BeamSearch."""
        self.model = model
        self.composition = None
        self.seq_length = seq_length

    def predict(self, start_input: np.ndarray, nr_measures: int) -> np.ndarray:
        """Predict some number of measures, given a start input."""
        self.composition = start_input

        for measure in range(nr_measures):
            print(measure)
            self._predict_one_measure()
        return self.composition[:, self.seq_length :]

    def _predict_one_measure(self) -> None:
        """Predict one measure.

        This corresponds to one sentence in a NLP task. One measure is a
        closed unit after which a new one begins. The created predictions
        will be appended to the `composition` attribute.
        """
        cumulated_duration = 0
        while cumulated_duration < 48:
            value, duration = self._predict_one_step()
            self._append_prediction_to_composition(value, duration, cumulated_duration)
            cumulated_duration += duration

    def _predict_one_step(self):
        """Predict one step.

        :return: The value and the duration that the network predicted.
        """
        reformatted_input = list(
            np.expand_dims(self.composition[:, -self.seq_length :], 1)
        )
        predictions = self.model.predict(reformatted_input)
        value, duration = [np.argmax(p) for p in predictions]
        return value, duration

    def _append_prediction_to_composition(
        self, value: int, duration: int, offset: int
    ) -> None:
        """Append the prediction to the composition.

        :param value: The value to append.
        :param duration: The duration to append.
        :param offset: The offset to append.
        """
        new_value = np.expand_dims([value, duration, offset], axis=1)
        self.composition = np.hstack([self.composition, new_value])
