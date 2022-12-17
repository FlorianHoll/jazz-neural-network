"""Implementation of a candidate for Beam Search."""
import numpy as np


class BeamSearchCandidate:
    """One candidate for Beam Search.

    :param value: The value, i.e. the "history" of the candidate.
    :param log_probability: The log probability of the candidate.
    """

    def __init__(self, value: np.ndarray, log_probability: float) -> None:
        """Initialize the candidate."""
        self.value = value
        self.log_probability = log_probability

    @property
    def _current_offset(self):
        """Return the current offset, i.e. before a value is predicted."""
        return self.value[-2, -1]

    @property
    def finished(self):
        """Indicate if the candidate has reached the end of the measure."""
        return self._current_offset == 0

    @property
    def value(self):
        """Return the value of the candidate, i.e. its history and input."""
        return self._value

    @value.setter
    def value(self, new_value: np.ndarray):
        """Set a new value if the input is sensible."""
        if not isinstance(new_value, np.ndarray):
            raise TypeError(
                f"A {new_value.__class__.__name__} was passed as the value for "
                f"a BeamSearchCandidate. Only numpy arrays are allowed."
            )
        self._value = new_value

    @property
    def log_probability(self):
        """Return the log probability of the candidate."""
        return self._log_probability

    @log_probability.setter
    def log_probability(self, new_value: float):
        """Set the log probability (if the input is sensible)."""
        if new_value > 0:
            raise ValueError(
                f"A log probability of {new_value} was passed. Since the log "
                f"probability can only take values <= 0, this is not sensible."
            )
        self._log_probability = new_value

    def __gt__(self, other):
        """Define greater than comparison with other objects (needed for sorting)."""
        if isinstance(other, BeamSearchCandidate):
            return self.log_probability > other.log_probability
        raise TypeError(
            "A comparison between a BeamSearchCandidate and a "
            f"{other.__class__.__name__} occurred. This should not occur."
        )

    def __lt__(self, other):
        """Define less than comparison with other objects (needed for sorting)."""
        if isinstance(other, BeamSearchCandidate):
            return self.log_probability < other.log_probability
        raise TypeError(
            "A comparison between a BeamSearchCandidate and a "
            f"{other.__class__.__name__} occurred. This should not occur."
        )

    def __len__(self):
        """Define length attribute (needed for checks)."""
        return self.value.shape[1]

    def __repr__(self):
        """Define the representation."""
        return (
            f"Candidate with length {len(self)} and "
            f"log prob {round(self.log_probability, 2)}."
        )
