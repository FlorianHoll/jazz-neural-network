"""A beam search candidate."""
from typing import Union

import numpy as np
import pandas as pd
from tensorflow import keras

from rnn.music import Note


class BeamSearchCandidate:
    """One candidate in the BeamSearch algorithm.

    :param value: the values of the candidate (i.e. notes, duration, offsets).
    :param prob: The probability of the candidate.
    :param seq_length: The sequence length that the NN expects.
    """

    MEASURE_LENGTH = 48

    def __init__(self, value: np.array, prob: float, seq_length: int):
        """Initialize the candidate."""
        self.value = value
        self.value = value  # TODO: Implement setter and getter methods
        self.prob = prob  # TODO: Implement stter and getter
        self.seq_length = seq_length

    def _clip_end_of_measure(self, durations: np.array) -> np.array:
        """Clip last note if it exceeds the end of the measure.

        :param durations: the durations of the candidate.
        :return clipped duration
        """
        current_length = np.sum(durations)
        if current_length > self.MEASURE_LENGTH:
            durations[-1] -= current_length - self.MEASURE_LENGTH
        return durations

    def _format_prediction_input(self):
        """Refactor the input into the needed format for the Neural Net."""
        prediction_input = list(self.value[:, -self.seq_length :])
        prediction_input = [var.reshape(1, self.seq_length) for var in prediction_input]
        return prediction_input

    def get_length(self):
        """Get the number of iterations the candidate went through until this point."""
        return self.value[:, self.seq_length :].shape[-1]

    def _get_timepoint(self):
        """Get the number of beats of the candidate (needed to get nearest chord)."""
        return np.sum(self.value[:, self.seq_length :])

    def _get_next_offset(self):
        """Get the offset of the chord/ note to be predicted."""
        measure_length = self._get_measure_length()
        next_offset = self.value[-1, -1] + self.value[-2, -1]
        return next_offset % measure_length

    def _get_curr_chord(self, chords: pd.DataFrame):
        """Get the chord that the current note will be played over.

        :param chords: The chords as a DataFrame.
        :return: np array of shape (4,) containing the four notes of the chord.
        """
        # TODO: Still necessary?
        offset = self._get_next_offset() / 12
        relevant_chords = chords.loc[
            (chords.measure == self.measure) & (chords.offset <= offset)
        ]
        min_diff = (relevant_chords.offset - offset).argmin()
        chord = relevant_chords.reset_index().loc[min_diff, ["c1", "c2", "c3", "c4"]]
        chord = [np.array([[c]]) for c in chord]
        return chord

    def _get_whole_nn_input(self, chords: Union[None, pd.DataFrame] = None) -> list:
        """Get the input ready for the neural net.

        :param chords: None if harmony is being written; pd.DataFrame if
            melody is being written.
        """
        prediction_params = self._format_prediction_input()
        if self.model_type.startswith("melody"):
            chord_input = self._get_curr_chord(chords)
            prediction_params = (
                [prediction_params[0]] + chord_input + prediction_params[1:]
            )
        return prediction_params

    def is_finished(self):
        """Return boolean indicator if candidate is finished."""
        next_offset = self._get_next_offset()
        return next_offset == 0

    def _get_values_and_probabilities(
        self, model_predictions: list, k: int
    ) -> [np.array, np.array]:
        """Get values and probabilities from model predictions.

        :param model_predictions: The output from the model.predict() method.
        :return the values and probabilites.
        """
        values = [np.argsort(p)[-k:] for p in model_predictions]
        values[-1] = self._clip_end_of_measure(values[-1])

        probabilities = [m[values[i]] for i, m in enumerate(model_predictions)]

        return np.array(values), np.array(probabilities)

    def _calculate_probabilities(
        self, probabilities: np.array, prob_scaling: float, weighting: list
    ) -> np.array:
        """
        Transform the 'raw' probabilties with the specified method.

        :param probabilities: array with the probabilities of the outputs.
        :param prob_scaling: prob_scaling factor.
        :param weighting: Weighting of the probabilities.
        :return: the transformed probabilities.
        """
        if prob_scaling < 1:
            probabilities = (
                np.dot(weighting, np.power(probabilities, prob_scaling)) * self.prob
            )
        else:
            probabilities = np.dot(weighting, np.log(probabilities)) + self.prob

        return probabilities

    def predict_next_k_best(
        self,
        k: int,
        model: keras.models.Model,
        candidates: list,
        prob_scaling: float,
        weighting: list,
    ):
        """Predict k best next candidates for a candidate by running RNN.

        :param k: How many candidates to create
        :param model: The trained neural network
        :param candidates: list to append new candidates to
        :param chords: see above
        :param prob_scaling: scaling of resulting probabilities
        :param weighting: weighting of value vs. duration.
        """
        nn_input = self._get_whole_nn_input(chords=chords)

        model_predictions = [m.flatten() for m in model.predict(nn_input)]

        values, probabilities = self._get_values_and_probabilities(model_predictions, k)

        probabilities = self._calculate_probabilities(
            probabilities, prob_scaling, weighting
        )

        shape_of_new_input = self.value.shape[0]
        for i in range(k):
            new_input = np.append(values[:, i], self._get_next_offset()).reshape(
                shape_of_new_input, 1
            )
            new_value = np.hstack([self.value, new_input])

            new_candidate = BeamSearchCandidate(
                value=new_value,
                prob=probabilities[i],
                seq_length=self.seq_length,
                model_type=self.model_type,
                measure=self.measure,
            )

            candidates.append(new_candidate)
