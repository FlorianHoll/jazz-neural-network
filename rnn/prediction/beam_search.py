"""Implementation of Beam Search for the Jazz Recurrent Neural Network.

Beam Search is an algorithm for machine learning models that return
probabilities (such as Neural nets) that takes into account not only
the most probable candidate (as in Greedy Search), but k candidates
that are evaluated at each time step. This potentially leads to
predictions that might make more sense overall although they were not
the most probable one initially.

Beam Search is often used in NLP problems where each word is one
network prediction and the search lasts until the sentence is finished.
Here, each prediction step is one note and the search lasts until one
measure is finished.
"""
import keras
import numpy as np

from rnn.prediction.beam_search_candidate import BeamSearchCandidate


class BeamSearch:
    """Beam Search.

    :param model: The model to predict from with weights already loaded.
    :param k: The number of candidates to take into account at each time step.
        k=1 == greedy search; for any k > 1, this is beam search.
    :param alpha: The length normalization scaler.
    :param seq_length: The length of one sequence passed to the neural net.
    :param weighting: Optionally, a list of weightings for each of the model's outputs.
        Note that the durations cannot be weighted.
    """

    def __init__(
        self,
        model: keras.Model,
        k: int = 3,
        alpha: float = 0.7,
        seq_length: int = 8,
        weighting: list[float, float] = None,
    ) -> None:
        """Initialize the BeamSearch."""
        self.model = model
        self.k = k
        self.alpha = alpha
        self.weighting = weighting
        self.composition = None
        self.seq_length = seq_length

    def predict(self, start_input: np.ndarray, nr_measures: int = 32) -> np.ndarray:
        """Predict some number of measures, given a start input."""
        self.composition = start_input

        for measure in range(nr_measures):
            print(measure)
            self._predict_one_measure()
        return self.composition

    def _predict_one_measure(self) -> None:
        """Predict one measure.

        This corresponds to one sentence in a NLP task. One measure is a
        closed unit after which a new one begins. The created predictions
        will be appended to the `composition` attribute.
        """
        start_input = self.composition[-self.seq_length :]
        candidates = [BeamSearchCandidate(start_input, 0.0)]
        finished_candidates = []
        while True:
            candidates = self._predict_one_step(candidates)
            for candidate in candidates:
                if candidate.finished:
                    finished_candidates.append(candidate)
                    candidates.remove(candidate)
            if len(candidates) == 0:
                break

    def _predict_one_step(
        self, current_candidates: list[BeamSearchCandidate]
    ) -> list[BeamSearchCandidate]:
        """Predict one Beam Search step.

        This means that for each the n active candidates, the k best candidates
        are created. Of all n*k candidates after this step, the k best ones are
        chosen. These are returned as the result of the Beam Search step.

        :param current_candidates: The active candidates; can be k, but does not
            have to be (if some candidates are already finished for example).
        :return: The k best candidates for the search step.
        """
        new_candidates = []
        for candidate in current_candidates:
            new_candidates += self._get_k_best_candidates_for_candidate(candidate)
        candidates = self._choose_k_best_candidates(new_candidates)
        return candidates

    def _get_k_best_candidates_for_candidate(
        self, candidate: BeamSearchCandidate
    ) -> list[BeamSearchCandidate]:
        """Get the k best candidates for a candidate.

        This means that the candidate's value is used to predict potential new values;
        of these, the k most probable ones are used.

        :param candidate: The candidate.
        :return: The k best candidates (i.e. next values) for this candidate.
        """
        k_best_candidates = []
        values, probabilities = self._get_values_and_probabilities(candidate)
        for i in range(self.k):
            k_best_candidates.append(
                BeamSearchCandidate(
                    self._reformat_values(candidate, values[:, i]), probabilities[i]
                ),
            )
        return k_best_candidates

    def _get_values_and_probabilities(self, candidate):
        """Predict from Neural Net and return the most probable values.

        :param candidate: The candidate for which the next values shall be predicted.
        :return: The most probable values and their respective (log) probabilities.
        """
        # Reformat input to suit the model.
        reformatted_input = list(
            np.expand_dims(candidate.value[:, -self.seq_length :], 1)
        )
        predictions = self.model.predict(reformatted_input)

        duration_value = np.argsort(predictions[-1]).flatten()[-1]

        # Extract values and probabilities.
        values = np.array(
            [np.argsort(p).flatten()[-self.k :] for p in predictions[:-1]]
        )
        values = np.vstack([values, np.repeat(duration_value, self.k)])
        probabilities = np.array(
            [np.sort(p).flatten()[-self.k :] for p in predictions[:-1]]
        )

        # Calculate the log probabilities for each value pair; optionally weighted
        #   with the given weighting:
        if self.weighting:
            probabilities = (
                np.dot(np.array(self.weighting), np.log(probabilities))
                + candidate.log_probability
            )
        else:
            probabilities = (
                np.sum(np.log(probabilities), axis=0) + candidate.log_probability
            )

        return values, probabilities

    def _choose_k_best_candidates(self, candidate_list: list[BeamSearchCandidate]):
        """Choose the k best candidates.

        Since the BeamSearchCandidate class implements the __ge__ and __le__ methods,
        this comes down to sorting the list of candidates.

        :param candidate_list: The list with the finished candidates to choose the k
            best ones from.
        :return: The k best candidates.
        """
        return sorted(candidate_list)[-self.k :]

    @staticmethod
    def _reformat_values(candidate: BeamSearchCandidate, value: tuple[float, float]):
        """Reformat the values for the next candidate.

        :param candidate: The candidate.
        :param value: The value for the candidate, i.e. its value and the duration.
        :return: A numpy array that appends the new values to the ones already present.
        """
        previous_values = candidate.value
        value, duration_value = value
        last_offset = previous_values[-1, -1]
        new_offset = duration_value + last_offset
        if new_offset > 48:
            duration_value = new_offset - new_offset % 48
            new_offset = 0
        new_value = np.expand_dims([value, duration_value, new_offset], axis=1)
        return np.hstack([candidate.value, new_value])
