"""Musical elements classes.

All musical elements have some things in common. This fact will
be used to represent them all in one framework to parse the data
as a sensible input to the neural network.
"""
import abc
import warnings
from typing import Union

import numpy as np

from rnn.music.utils import chord_symbol_to_neural_net_representation
from rnn.music.utils import chord_type_to_compatible_chord
from rnn.music.utils import chord_type_to_numbers
from rnn.music.utils import functional_chord_notes_to_chord_symbol
from rnn.music.utils import midi_number_to_note_symbol
from rnn.music.utils import neural_net_representation_to_chord_symbol
from rnn.music.utils import neural_net_representation_to_note
from rnn.music.utils import note_symbol_to_number
from rnn.music.utils import note_to_neural_net_representation
from rnn.music.utils import pitch_heights_in_range


class MusicalElement:
    """Basic musical element.

    All musical elements have in common that are characterised by a
    duration and a offset in the measure they are played in.

    :param duration: The duration of the musical element (where one
        quarter note is represented as 12, a half note is 24, a whole
        note 48 etc.)
    :param offset: The offset of the element in the measure (on the
        same scale as the duration).
    """

    def __init__(
        self,
        duration: int = 12,
        offset: int = 0,
    ) -> None:
        """Initialize the MusicalElement."""
        self.duration = duration
        self.offset = offset

    @property
    def duration(self):
        """Get the duration."""
        return self._duration

    @duration.setter
    def duration(self, new_value: int):
        """Set the duration to a new value after checking that the value is valid."""
        if new_value <= 0 or new_value > 48:
            raise ValueError(
                "The duration value trying to be set is outside of the plausible "
                "range for a note duration. Possible values are 1 - 48."
            )
        self._duration = new_value

    @property
    def offset(self):
        """Get the offset."""
        return self._offset

    @offset.setter
    def offset(self, new_value: int):
        """Set the offset to a new value after checking that the value is valid."""
        if new_value < 0 or new_value > 47:
            raise ValueError(
                "The offset value trying to be set is outside of the plausible "
                "range for a note offset. Possible values are 0 - 47."
            )
        self._offset = new_value

    @property
    @abc.abstractmethod
    def octave(self):
        """Get the octave of the element."""

    @property
    @abc.abstractmethod
    def symbol(self) -> str:
        """Get the element as a symbol."""

    @property
    @abc.abstractmethod
    def neural_net_representation(self):
        """Get the element in the representation for the neural net."""

    @classmethod
    @abc.abstractmethod
    def from_pitch_height(
        cls, pitch_height: Union[int, np.ndarray], duration: int, offset: int
    ):
        """Construct a musical element by giving a pitch height."""

    @classmethod
    @abc.abstractmethod
    def from_neural_net_representation(
        cls, neural_net_representation: int, duration: int, offset: int
    ):
        """Construct a musical element by giving a neural net representation."""

    @classmethod
    @abc.abstractmethod
    def from_symbol(cls, symbol: str, duration: int, offset: int):
        """Construct a musical element by giving a symbol."""

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
    """A rest element.

    Rest elements are special elements in that they do not
    have a pitch height; however, they still need to be
    represented in our neural net since they are an
    important part of music.
    """

    @classmethod
    def from_pitch_height(
        cls, pitch_height: int, duration: int, offset: int
    ) -> "RestElement":
        """
        Construct a RestElement from a pitch height.

        Since a RestElement does not have a pitch height, the parameter
        is ignored.
        """
        return cls(duration, offset)

    @classmethod
    def from_neural_net_representation(
        cls, neural_net_representation: int, duration: int, offset: int
    ) -> "RestElement":
        """Construct a RestElement from a neural net representation."""
        return cls(duration, offset)

    @classmethod
    def from_symbol(cls, symbol: str, duration: int, offset: int) -> "RestElement":
        """Construct a RestElement from a symbol."""
        return cls(duration, offset)

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

    def octave_down(self) -> "RestElement":
        """Transpose the rest element one octave down.

        Since transposing a rest element does not make
        sense, the instance is returned.
        """
        return self

    def octave_up(self) -> "RestElement":
        """Transpose the rest element one octave up.

        Since transposing a rest element does not make
        sense, the instance is returned.
        """
        return self

    def __eq__(self, other):
        """Define equality comparison.

        Two rests are the same if they occur in the same place
        of the measure, i.e. they have the same duration and same
        offset.
        """
        if isinstance(other, RestElement):
            return (self.duration == other.duration) & (self.offset == other.offset)
        return False

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
    """A note.

    A note has a corresponding MIDI value which is taken as the
    primary representation of the note since it is unambiguous.

    :param pitch_height: The pitch height, in MIDI representation
        (i.e. 60 = C4, 67 = G4, 72 = C5 etc.). Since the piano
        keyboard only ranges from MIDI pitches 21 - 108, only
        these values are accepted.
    :param duration: The duration, where 12 = quarter note, 24 =
        half note, 48 = whole note, etc.
    :param offset: The offset of the note in the measure (on the
        same scale as the duration).
    """

    def __init__(self, pitch_height: int, duration: int = 12, offset: int = 0) -> None:
        """Initialize the Note.

        In the initializer of the parent class, the setter checks
        will take place, leading to errors if the note is not
        initialized properly.
        """
        self.pitch_height = pitch_height
        super().__init__(duration, offset)

    @property
    def pitch_height(self):
        """Get the pitch height."""
        return self._pitch_height

    @pitch_height.setter
    def pitch_height(self, new_value: int):
        """Set the pitch height if the new value is valid."""
        if not pitch_heights_in_range(new_value):
            raise ValueError(
                "The given pitch height is outside the plausible range (21-108)."
            )
        self._pitch_height = new_value

    @classmethod
    def from_pitch_height(
        cls, pitch_height: int, duration: int = 12, offset: int = 0
    ) -> Union["Note", RestNote]:
        """
        Construct a note from a MIDI pitch height.

        Only MIDI pitch heights given as integers are accepted.

        :param pitch_height: The MIDI pitch height.
        :param duration: The duration of the note (where 12 = quarter note,
            48 = whole note, etc.).
        :param offset: The offset of the note (on the same scale as the duration).
        :return: A Note instance.
        """
        if pitch_height == 0:
            return RestNote(duration, offset)
        try:
            pitch_height = int(pitch_height)
        except ValueError:
            raise TypeError(
                "Wrong input type passed. Only integers are accepted as "
                "MIDI pitch heights."
            )
        return cls(pitch_height, duration, offset)

    @classmethod
    def from_neural_net_representation(
        cls, neural_net_representation: int, duration: int = 12, offset: int = 0
    ) -> Union["Note", RestNote]:
        """
        Construct a note from a neural net representation.

        Since the neural net output layer does not need to be unnecessarily
        large, it is not given values that are not possible or not played.
        This constructor wrapper constructor converts the neural net
        representation into the corresponding MIDI pitch and constructs
        the note with it.

        :param neural_net_representation: The neural net representation of
            the note (e.g. 12 = C4)
        :param duration: The duration of the note (where 12 = quarter note,
            48 = whole note, etc.).
        :param offset: The offset of the note (on the same scale as the duration).
        :return: A Note instance.
        """
        if neural_net_representation == 0:
            return RestNote(duration, offset)
        midi_pitch = neural_net_representation_to_note(neural_net_representation)
        return cls(midi_pitch, duration, offset)

    @classmethod
    def from_symbol(
        cls, symbol: str, duration: int = 12, offset: int = 0
    ) -> Union["Note", RestNote]:
        """
        Construct a note from a symbol.

        :param symbol: The note symbol, e.g. 'C4', 'Ab3', 'Bb5' etc.
        :param duration: The duration of the note (where 12 = quarter note,
            48 = whole note, etc.).
        :param offset: The offset of the note (on the same scale as the duration).
        :return:
        """
        if symbol == "Rest":
            return RestNote(duration, offset)
        try:
            octave = int(symbol[-1])
        except ValueError:
            raise ValueError(
                f"Handed a wrong note symbol ({symbol}). The symbol must contain"
                f"contain an indicator of the octave, e.g. 'C#4' or 'Ab5'."
            )
        note_symbol = symbol[:-1]
        pitch_height = note_symbol_to_number(note_symbol) + 12 * (octave + 1)
        return cls(pitch_height, duration, offset)

    @property
    def octave(self) -> int:
        """Get the octave of the note.

        This means a floor division by 12.
        Since 60 = C4, we need to subtract one to get the
        octave (e.g. 65 (=F4) // 12 = 5; 5 - 1 = 4).
        """
        return self.pitch_height // 12 - 1

    @property
    def symbol(self) -> str:
        """Get the representation as a symbol.

        For this, we need the root note, which is the modulus of 12.
        This is concatenated with the octave to obtain the final symbol.
        """
        root_note = midi_number_to_note_symbol(self.pitch_height % 12)
        return f"{root_note}{self.octave}"

    @property
    def neural_net_representation(self):
        """Return the neural net representation of the note.

        We do not need all notes in the neural net; the softmax
        in the output layer does not need implausible values; therefore,
        we will subtract 4 octaves so that only sensible values
        are given to the neural net in the first place.
        """
        return note_to_neural_net_representation(self.pitch_height)

    @property
    def hertz(self):
        """Return the Hz of the note.

        Hertz is a measure of how quickly the note's sound wave oscillates
        in one second.
        This is not needed in the neural network, but just implemented for fun.
        """
        one_note_hertz_change = np.power(2, 1 / 12)
        return 440 * np.power(one_note_hertz_change, self.pitch_height - 69)

    def _check_if_transposing_works(self, steps: int) -> None:
        """Check if the transposing works.

        This means checking if the new pitch height is still within
        the sensible range.
        """
        new_pitch_height = self.pitch_height + steps
        if not pitch_heights_in_range(new_pitch_height):
            raise ValueError(
                "Cannot transpose; resulting pitch height(s) values "
                "would be outside the playable range."
            )
        resulting_neural_net_repr = note_to_neural_net_representation(new_pitch_height)
        if resulting_neural_net_repr <= 0:
            warnings.warn(
                f"The note that is being transposed ({self}) will have a "
                f"neural net representation of < 0 ({resulting_neural_net_repr})"
                f"by transposing it by {steps} steps."
            )

    def transpose(self, steps: int):
        """Transpose the note."""
        self._check_if_transposing_works(steps)
        return Note.from_pitch_height(
            self.pitch_height + steps, self.duration, self.offset
        )

    def octave_down(self):
        """Transpose the note one octave down."""
        return self.transpose(-12)

    def octave_up(self):
        """Transpose the note one octave up."""
        return self.transpose(12)

    def __eq__(self, other) -> bool:
        """Define equality comparison.

        A note is the same note if it has the same MIDI pitch height,
        the same duration and the same offset. If a note has the same
        symbol (e.g. C#), but a different octave, the notes are not the
        same note - this is reflected in the MIDI pitch height. Therefore,
        it is used as the indicator to compare the two objects.
        """
        if isinstance(other, Note):
            return (
                (self.pitch_height == other.pitch_height)
                & (self.duration == other.duration)
                & (self.offset == other.offset)
            )
        return False

    def __repr__(self):
        """Represent the note with all relevant information."""
        return (
            f"Note {self.symbol} "
            f"(duration: {self.duration}, "
            f"offset: {self.offset})"
        )


class Chord(MusicalElement):
    """A chord.

    Since a chord consists of several note, the Chord class is structured
    around the idea that the chord is made up of individual Note instances.
    However, since the Note instances belong to a Chord, they have several
    attributes in common: The duration and offset have to be the same for
    all Note instances since all of them are in the chord.

    :param notes: The notes that the chord consists of (a list of Note
        instances). Since the duration and offset are given separately,
        duration and offsets of the individual notes will be overwritten.
    :param duration: The duration of the chord, where 12 = quarter note,
        24 = half note, 48 = whole note, etc.
    :param offset: The offset of the chord in the measure (on the same scale
        as the duration).
    """

    def __init__(self, notes: list[Note], duration: int = 12, offset: int = 0) -> None:
        """Initialize the Chord.

        In the initializer of the parents, the setter checks
        will take place - if the input is not valid, an error
        will be raised.
        """
        self.notes = notes
        super().__init__(duration, offset)
        self.pitch_height = np.array([note.pitch_height for note in notes])

    @property
    def duration(self):
        """Get the duration."""
        return self._duration

    @duration.setter
    def duration(self, new_value: int):
        """Overwrite setter method of base class.

        This is done since we want to overwrite all individual note's
        durations when setting the overall chord duration.
        If a non-plausible value should occur, the setter method of the
        note objects will raise an error.

        :param new_value: The new value to be set (integer between 1-48).
        """
        for note in self.notes:
            note.duration = new_value
        self._duration = new_value

    @property
    def offset(self):
        """Get the offset."""
        return self._offset

    @offset.setter
    def offset(self, new_value: int):
        """Overwrite setter method of base class.

        This is done because we want to overwrite all individual note's
        offsets when setting the overall chord offset.
        If a non-plausible value should occur, the setter method of the
        note objects will raise an error.

        :param new_value: The new value to be set (integer between 0-47).
        """
        for note in self.notes:
            note.offset = new_value
        self._offset = new_value

    @classmethod
    def from_pitch_height(
        cls, pitch_height: np.ndarray, duration: int = 12, offset: int = 0
    ) -> Union["Chord", "RestChord"]:
        """
        Construct a chord from an array of pitch heights.

        :param pitch_height: The pitch heights, i.e. an array of MIDI pitches.
            For example, np.r_[60, 63, 67, 70] = Cmin7; np.r_[60, 64, 67, 71] = Cmaj7.
        :param duration: The duration of the chord.
        :param offset: The offset of the chord.
        :return: A Chord instance.
        """
        if np.all(pitch_height == np.zeros(4)):
            return RestChord(duration, offset)
        # Overwrite duration and offset of the individual
        #   notes with the given parameter.
        notes = [
            Note.from_pitch_height(pitch, duration, offset) for pitch in pitch_height
        ]
        return cls(notes, duration, offset)

    @classmethod
    def from_neural_net_representation(
        cls, neural_net_representation: int, duration: int = 12, offset: int = 0
    ) -> Union["Chord", "RestChord"]:
        """
        Construct a chord from a neural net representation.

        The neural net representation converts chords into a unique
        number that then corresponds to the output neuron (e.g. 'Cmin7' -> 32).
        With this method, the process can be reversed: From a neural net
        prediction (e.g. 32), one can get the corresponding chord ('Cmin7').

        :param neural_net_representation: The neural net representation of the chord.
        :param duration: The duration of the chord, where 12 = quarter note, 24 = half
            note, 48 = whole note, etc.
        :param offset: The offset of the chord (on the same scale as the duration).
        :return: A Chord instance.
        """
        try:
            neural_net_representation = int(neural_net_representation)
        except ValueError:
            raise TypeError(
                "The neural net representation must be an integer between 0 and 60."
            )
        if neural_net_representation == 0:
            return RestChord(duration, offset)
        chord_symbol = neural_net_representation_to_chord_symbol(
            neural_net_representation
        )
        return cls.from_symbol(chord_symbol, duration, offset)

    @classmethod
    def from_symbol(
        cls, symbol: str, duration: int = 12, offset: int = 0
    ) -> Union["Chord", RestChord]:
        """
        Construct a chord from a chord symbol.

        :param symbol: The symbol. Possible symbols are composed by
            combining two elements:
            - A root symbol (C, C#, D, ..., B)
            - a chord type (only possible: 'min7', 'dom7', 'maj7', 'dim7')
            These two elements have to be separated by a space.
            For example, 'A maj7' is a valid chord symbol, 'A', 'A major 'Amaj7'
            are not valid chord symbols.
            The restriction to have only four chord types actually represents
            the way that jazz musicians think about these chord: In the lead
            sheet, these are often the only "instructions" and the musician has
            to come up with the exact voicing, extensions etc. to play; therefore,
            the network does not have to learn every possible chord extension
            because the extension are in almost all cases NOT part of the written
            harmony (which the network is fed and is supposed to learn).
        :param duration: The duration of the chord, where 12 = quarter note,
            24 = half note, 48 = whole note, etc.
        :param offset: The offset of the note (on the same scale as the duration).
        :return: A chord instance.
        """
        # Check if the chord is actually a RestChord.
        if symbol == "N.C.":
            return RestChord(duration, offset)

        # Split the symbol into the relevant information.
        symbol_length = 2 if symbol[1] in ("#", "b") else 1
        root = symbol[:symbol_length]
        chord_type = symbol[symbol_length:].replace(" ", "")
        chord_type = chord_type_to_compatible_chord(chord_type)
        # By default, the chord will have a root in the fourth octave.
        root_symbol = f"{root}{4}"

        # Construct the chord by getting the pitch height of the
        #   root note and then adding the amount of half tones
        #   needed to produce the functionality of the chord, e.g.
        #   [0, 3, 7, 10] for a min7 chord, [0, 4, 7, 11] for a maj7
        #   chord etc.
        pitch_heights_from_root = chord_type_to_numbers(chord_type)
        root_pitch_height = Note.from_symbol(root_symbol).pitch_height
        pitch_heights = root_pitch_height + pitch_heights_from_root

        # Construct the notes and return the chord.
        notes = [
            Note.from_pitch_height(pitch_height, duration, offset)
            for pitch_height in pitch_heights
        ]
        return cls(notes, duration, offset)

    @property
    def root_note(self):
        """Get the root note (as a Note instance)."""
        return self.notes[0]

    @property
    def chord_type(self):
        """Get the chord type symbol."""
        return self.symbol.split(" ")[-1]

    @property
    def octave(self):
        """Get the octaves of all notes."""
        return [note.octave for note in self.notes]

    @property
    def symbol(self) -> str:
        """Get the chord symbol."""
        root = self.root_note.pitch_height
        functional_notes = np.array([note.pitch_height - root for note in self.notes])
        chord_type = functional_chord_notes_to_chord_symbol(functional_notes)
        return f"{self.root_note.symbol[:-1]} {chord_type}"

    @property
    def neural_net_representation(self) -> int:
        """Get the neural net representation."""
        return chord_symbol_to_neural_net_representation(self.symbol)

    @property
    def pitch_neural_net_representation(self) -> np.ndarray:
        """Get the neural net representation of the notes."""
        return np.array([note.neural_net_representation for note in self.notes])

    def transpose(self, steps: int):
        """Transpose the whole chord."""
        transposed_notes = [note.transpose(steps) for note in self.notes]
        return Chord(transposed_notes, self.duration, self.offset)

    def octave_down(self):
        """Transpose the whole chord ."""
        return self.transpose(-12)

    def octave_up(self):
        """Test."""
        return self.transpose(12)

    def __eq__(self, other):
        """Define equality comparison.

        A chord is the same if the notes that it consists of are the same.
        Therefore, we simply compare if all notes are the same. If so, the
        chords are the same. Since the notes incorporate the duration and
        the offset of the chord, this doesn't have to be checked separately.
        """
        if isinstance(other, Chord):
            return all(
                chord_note == other_note
                for chord_note, other_note in zip(self.notes, other.notes)
            )
        return False

    def __repr__(self):
        """Represent the chord with all relevant information."""
        return (
            f"Chord {self.symbol} (duration: {self.duration}, "
            f"offset: {self.offset})"
        )
