"""Musical elements classes.

All musical elements have some things in common. This fact will
be used to represent them all in one framework to parse the data
as a sensible input to the neural network.
"""
import abc
from typing import Union

import numpy as np

from rnn.music.utils import octave_in_range

BASSNOTE_RANGE = [40, 64]
NOTE_RANGE = [21, 108]
MELODY_NOTE_RANGE = [48, 108]
OCTAVE_RANGE = [0, 8]
DURATION_RANGE = [3, 48]
OFFSET_RANGE = [0, 45]

SYMBOL_MISSING = "In order to transform to notes, you must hand a symbol."
NUMBER_MISSING = "In order to transform to a symbol, you must hand note numbers."


class MusicalElement:
    """Basic musical element."""

    def __init__(
        self,
        element: Union[str, int, np.ndarray],
        duration: int = 12,
        offset: int = 0,
        octave: int = 4,
    ) -> None:
        """Initialize the MusicalElement."""
        self._initial_representation = element
        self.duration = duration
        self.offset = offset
        self.octave = octave
        if isinstance(element, str):
            if not octave_in_range(octave):
                raise ValueError(
                    "The given octave value is outside the plausible "
                    "range; 0-8 are acceptable values."
                )

    @property
    @abc.abstractmethod
    def pitch_height(self):
        """Get the pitch height of the element."""

    @property
    @abc.abstractmethod
    def symbol(self) -> str:
        """Get the element as a symbol."""

    @property
    @abc.abstractmethod
    def neural_net_representation(self):
        """Get the element in the representation for the NN."""

    @abc.abstractmethod
    def transpose(self, steps):
        """Transpose the element."""

    @abc.abstractmethod
    def octaves_down(self, num_octaves: int = 1):
        """Transpose the element a given amount of octaves down."""

    @abc.abstractmethod
    def octaves_up(self, num_octaves: int = 1):
        """Transpose the element a given amount of octaves up."""


class RestElement(MusicalElement):
    """A rest element."""

    def __init__(
        self,
        duration: int = 12,
        offset: int = 0,
    ) -> None:
        """Initialize the RestElement."""
        super().__init__(element="Rest", duration=duration, offset=offset, octave=0)

    @property
    def pitch_height(self) -> Union[int, np.ndarray]:
        """Return the pitch height.

        0 is used here as a placeholder to indicate
        that the element does not have a pitch height.
        """
        return 0

    @property
    def symbol(self) -> str:
        """Return the symbol."""
        return "Rest"

    @property
    def neural_net_representation(self) -> Union[int, np.ndarray]:
        """Return the neural net representation.

        Will always be 0 to indicate that this is a rest.
        """
        return 0

    def transpose(self, steps):
        """Transpose the rest element.

        Since transposing a rest element does not make
        sense, the instance is returned.
        """
        return self

    def octaves_down(self, num_octaves: int = 1):
        """Transpose the rest element one octave down.

        Since transposing a rest element does not make
        sense, the instance is returned.
        """
        return self

    def octaves_up(self, num_octaves: int = 1):
        """Transpose the rest element one octave up.

        Since transposing a rest element does not make
        sense, the instance is returned.
        """
        return self

    def __repr__(self):
        """Represent the rest element with all relevant information."""
        return f"Rest (duration: {self.duration}, offset: {self.offset})"


class RestNote(RestElement):
    """A rest note."""


class RestChord(RestElement):
    """A rest chord."""

    @property
    def pitch_height(self):
        """Return the pitch height.

        0 is used as a placeholder to indicate that the chord does
        not have a pitch height; however, since a chord (in our
        framework) always consists of four notes, four zeros are
        returned to ensure compatibility with other classes.
        """
        return np.zeros(4)

    @property
    def symbol(self):
        """Return the symbol.

        'N.C.' stands for 'No Chord', indicating that no chord is present
        here.
        """
        return "N.C."

    def __repr__(self):
        """Represent the rest chord with all relevant information."""
        return f"N.C. (duration: {self.duration}, offset: {self.offset})"
