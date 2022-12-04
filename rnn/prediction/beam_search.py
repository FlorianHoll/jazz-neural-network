"""Beam Search Implementation for writing a melody over a given chord progression."""
import keras.models
import numpy as np

from rnn.prediction.beam_search_candidate import BeamSearchCandidate


class BeamSearch:
    """Conduct beam search either for harmony or for melody generation.

    :param model: The model with loaded weights.
    :param k: Beam width, i.e. number of considered candidates at each steps.
        If k == 1, then greedy search is performed because no other candidates
        are considered. If k > 1, beam search is performed with k candidates
        considered at each time step.
    :param alpha: alpha for length normalization:
        1 / length(candidate) ** alpha * prob(candidate)
        alpha = 1 --> complete length normalization
        alpha = 0 --> no length normalization
    :param prob_scaling: own method, so handle carefully!
        probabilities of model are exponentiated with this factor
        prob_scaling = 1 --> no scaling at all
        prob_scaling < 1 --> favors smaller probabilities
        i.e. less 'obvious' candidates can be favored to some degree
    :param weighting: weighting of the outputs. Defaults to 1 for all outputs.
    :param nr_measures: The number of measures to write.
    """

    def __init__(
        self,
        model: keras.models.Model,
        k: int = 3,
        alpha: float = 0.7,
        prob_scaling: float = 0.1,
        weighting: list = None,
        nr_measures: int = 32,
    ):
        """Initialize the BeamSearch."""
        if weighting is None:
            weighting = [1, 1]
        self.model = model
        self.k = k
        self.alpha = alpha
        self.scaling = prob_scaling
        self.weighting = weighting
        self.nr_measures = nr_measures
        self.seq_length = self.model.input_shape[0][1]
        self.composition = None

    @staticmethod
    def _continue_searching(cand_list):
        """Return False if all candidates in candidate list are finished."""
        return len(cand_list) != 0

    def _choose_k_best(self, candidate_list: list):
        """Choose k best of a list of candidates.

        This is done based on the probabilities of the candidates - the ones with the
        highest probabilities are taken as the best ones.

        :param candidate_list: The list with all candidates
        :return list with k best candidates.
        """
        k_best = sorted(candidate_list, key=lambda x: x.prob, reverse=True)[: self.k]
        return k_best

    def _choose_final_cand(self, finished_candidates: list[BeamSearchCandidate]):
        """Implement length normalization.

        :param finished_candidates: list of finished candidates to choose frmo
        :return: list with only the final candidate.
        """
        final_cand = sorted(
            finished_candidates,
            key=lambda x: 1 / x.get_length() ** self.alpha * x.prob,
            reverse=True,
        )[:1][0]
        final_cand.value = final_cand.value[:, self.seq_length :]  # TODO: More elegant?
        return final_cand

    def compose(self, start_input: np.array):
        """Implement beam search algorithm. Parameters are given by class."""
        self.composition = start_input

        for i in range(self.nr_measures):

            # List for candidates who reached end of measure
            finished_candidates = []

            # Initiate first candidate as final candidate from last measure
            initial_prob = 1.0 if self.scaling < 1 else 0.0  # TODO: Why?
            first_candidate = BeamSearchCandidate(
                value=self.composition[:, -self.seq_length :],
                prob=initial_prob,
                seq_length=self.seq_length,
                measure=i,
            )
            candidates = [first_candidate]

            # Beam search until all candidates are finished.
            while True:
                new_candidates = []
                # Get k best candidates for each candidate in list

                for cand in candidates:
                    cand.predict_next_k_best(
                        k=self.k,
                        model=self.model,
                        candidates=new_candidates,
                        prob_scaling=self.scaling,
                        weighting=self.weighting,
                    )

                # Choose k best ones of the newly created candidates
                candidates = self._choose_k_best(new_candidates)

                finished_candidates += [c for c in candidates if c.is_finished()]
                candidates = [c for c in candidates if not c.is_finished()]

                if len(candidates) == 0:
                    break

            final_cand = self._choose_final_cand(finished_candidates)

            self.composition = np.hstack([self.composition, final_cand.value])

        # Remove input from beginning so that only the composed part remains
        return self.composition[:, self.seq_length :]
