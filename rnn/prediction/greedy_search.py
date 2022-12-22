"""
Implementation of Greedy Search for the Jazz Recurrent Neural Network.

Greedy Search is the easiest way of predicting from a Neural network; the most
probable value at each step is simply taken to be the next value.
"""
from typing import List

import keras
import numpy as np

from rnn.music import Chord


class GreedySearch:
    """Greedy Search.

    :param model: The model to predict from with weights already loaded.
    :param seq_length: The length of one sequence passed to the neural net.
    """

    def __init__(
        self,
        model: keras.Model,
        start_input: np.ndarray,
        nr_measures: int,
        chords: np.ndarray = None,
        seq_length: int = 8,
    ) -> None:
        """Initialize the BeamSearch."""
        self.model = model
        self.composition = start_input
        self.seq_length = seq_length
        self.nr_measures = nr_measures
        self.chords = chords

    @property
    def summed_duration_of_predicted_notes(self):
        """Return the cumulated duration of all predicted notes.

        This is needed for the tracking of the chords (the cumulated duration
        basically works like a cursor that shows where we are in the harmony
        right now. Based on this, the current chord that a note is played over
        can be found out).
        """
        return np.sum(self.composition[1, self.seq_length :])

    def predict(self) -> np.ndarray:
        """Predict some number of measures, given a start input."""
        for measure in range(self.nr_measures):
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
        reformatted_input = self._reformat_input()
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
        self.composition = np.hstack([self.composition, new_value])  #

    def _reformat_input(self):
        """Reformat the input for the neural net.

        Expanding the dimension is needed to get the arrays into the correct format.
        :return: The formatted input.
        """
        return list(np.expand_dims(self.composition[:, -self.seq_length :], 1))


class HarmonyGreedySearch(GreedySearch):
    """Harmony Greedy Search (just the basic implementation will do)."""


class MelodyGreedySearch(GreedySearch):
    """Melody Greedy Search (needs the functionality to follow the chords)."""

    def _reformat_input(self):
        """Reformat the input for the neural net.

        In the case of the melody, this means to additionally keep track of the
        chords that the next note will be played over.
        """
        previous_notes = list(
            np.expand_dims(self.composition[:, -self.seq_length :], 1)
        )
        relevant_chords = self.chords[
            0, self.chords[3, :] <= self.summed_duration_of_predicted_notes
        ]
        chord = Chord.from_neural_net_representation(relevant_chords[-1])
        current_chord = list(np.expand_dims(chord.pitch_neural_net_representation, 1))
        return previous_notes + current_chord
