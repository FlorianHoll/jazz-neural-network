"""Musical elements classes.

All musical elements have some things in common. This fact will
be used to represent them all in one framework to parse the data
as a sensible input to the neural network.
"""
import abc
from typing import Union

import numpy as np

from rnn.music.utils import MIDI_NUMBER_TO_NOTE_SYMBOL
from rnn.music.utils import NOTE_SYMBOL_TO_NUMBER
from rnn.music.utils import octave_in_range
from rnn.music.utils import pitch_height_in_range

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
    ) -> None:
        """Initialize the MusicalElement."""
        self._initial_representation = element
        self.duration = duration
        self.offset = offset

    @property
    @abc.abstractmethod
    def octave(self):
        """Get the octave of the element."""

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
    def transpose(self, steps: int):
        """Transpose the element."""

    @abc.abstractmethod
    def octave_down(self):
        """Transpose the element one octave down."""

    @abc.abstractmethod
    def octave_up(self):
        """Transpose the element one octave up."""


class RestElement(MusicalElement):
    """A rest element."""

    def __init__(
        self,
        duration: int = 12,
        offset: int = 0,
    ) -> None:
        """Initialize the RestElement."""
        super().__init__(element="Rest", duration=duration, offset=offset)

    @property
    def octave(self) -> int:
        """Return the octave.

        0 is used here as a placeholder to indicate
        that the element does not have an octave.
        """
        return 0

    @property
    def pitch_height(self) -> int:
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
    def neural_net_representation(self) -> int:
        """Return the neural net representation.

        Will always be 0 to indicate that this is a rest.
        """
        return 0

    def transpose(self, steps: int) -> "RestElement":
        """Transpose the rest element.

        Since transposing a rest element does not make
        sense, the instance is returned.
        """
        return self

    def octave_down(self, num_octaves: int = 1) -> "RestElement":
        """Transpose the rest element one octave down.

        Since transposing a rest element does not make
        sense, the instance is returned.
        """
        return self

    def octave_up(self, num_octaves: int = 1) -> "RestElement":
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


class Note(MusicalElement):
    """A note."""

    @property
    def symbol(self) -> str:
        """Get the symbol of the note."""
        return self._initial_representation

    @property
    def octave(self):
        """Get the octave of the note."""
        return int(self.symbol[-1])

    @property
    def pitch_height(self) -> int:
        """Get the MIDI pitch height of the note."""
        pitch_height = NOTE_SYMBOL_TO_NUMBER[self.symbol[:-1]] + 12 * (self.octave + 1)
        if pitch_height_in_range(pitch_height):
            return pitch_height
        else:
            raise ValueError("The given note is outside the sensible range.")

    @property
    def neural_net_representation(self):
        """Return the neural net representation of the note."""
        return self.pitch_height - 48

    def _check_if_transposing_works(self, steps: int) -> None:
        """Check if the transposing works.

        This means checking if the new pitch height is still within
        the sensible range.
        """
        new_pitch_height = self.pitch_height + steps
        if not pitch_height_in_range(new_pitch_height):
            raise ValueError(
                "Cannot transpose; resulting pitch height values would "
                "be outside the playable range."
            )

    def transpose(self, steps: int):
        """Transpose the note."""
        self._check_if_transposing_works(steps)
        return MidiNote(self.pitch_height + steps)

    def octave_down(self):
        """Transpose the note one octave down."""
        return self.transpose(-12)

    def octave_up(self):
        """Tranpose the note one octave up."""
        return self.transpose(12)

    def __repr__(self):
        """Represent the note with all relevant information."""
        return (
            f"Note {self.symbol} (duration: {self.duration}, " f"offset: {self.offset})"
        )


class MidiNote(Note):
    """A Note in MIDI pitch height format."""

    @property
    def pitch_height(self) -> int:
        """Get the pitch height of the note."""
        return self._initial_representation

    @property
    def octave(self) -> int:
        """Get the octave of the note."""
        return self.pitch_height // 12 - 1

    @property
    def symbol(self) -> str:
        """Get the representation as a symbol."""
        pitch = self._initial_representation
        while pitch not in range(12):
            pitch -= 12
        root_note = MIDI_NUMBER_TO_NOTE_SYMBOL[pitch]
        return f"{root_note}{self.octave}"
