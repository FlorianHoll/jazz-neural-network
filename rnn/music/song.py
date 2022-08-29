"""A song in the training corpus."""
from typing import List
from typing import Union

import bs4.element
import numpy as np
from bs4 import BeautifulSoup

from rnn.music.musical_elements import Chord
from rnn.music.musical_elements import Note
from rnn.music.musical_elements import RestChord
from rnn.music.musical_elements import RestNote
from rnn.music.utils import CHORD_TYPE_TO_COMPATIBLE_CHORD
from rnn.music.utils import SHARPS_TO_KEY_SIGNATURE_SYMBOL


class Song:
    """A song that will be used to train the neural network."""

    def __init__(self, filename: str):
        """Initialize the song."""
        with open(filename, "r") as file:
            data = file.read()
        self.raw_data = BeautifulSoup(data, "xml")

        self.melody_representation = []
        self.harmony_representation = []

    @property
    def key_signature(self):
        """Get the key signature of the whole song."""
        key_signature_in_fifths_from_c = int(
            self.raw_data.find("key").find("fifths").string
        )
        return SHARPS_TO_KEY_SIGNATURE_SYMBOL[key_signature_in_fifths_from_c]

    def parse(self):
        """Parse the song from the xml representation."""
        measures = self.raw_data.findChildren("measure")
        for measure in measures:
            self._parse_one_measure(measure)
        return self.melody_neural_net_representation

    def _parse_one_measure(self, measure):
        elements = measure.findChildren(["note", "harmony"])  # find all notes.
        # The offset is set to 0 initially and will be updated iteratively.
        offset = 0
        for element in elements:
            # First, find out if the element is a note or a harmony symbol.
            if element.name == "note":
                note = self._parse_one_note(element, offset)
                self.melody_representation.append(note)
                # The offset must be updated to keep track of the position
                #   of the measure that is currently being parsed.
                offset += note.duration
            else:
                harmony = self._parse_one_harmony_symbol(element, offset)
                self.harmony_representation.append(harmony)

    @staticmethod
    def _parse_one_note(note: bs4.element.Tag, offset: int) -> Union[Note, RestNote]:
        """Parse one note and return it."""
        # The durations as written in the .xml have to be multiplied
        #   by four to represent our time grid.
        note_duration = int(note.find("duration").string)

        # If the note is NOT a rest, it will have the 'step'
        #   and 'octave' attributes; however, if it IS a rest,
        #   an AttributeError will be raised because the attribute
        #   does not exist.
        try:
            note_symbol = note.find("step").string
            note_octave = note.find("octave").string
            try:
                # If the note is altered (i.e. raised or flattened),
                #   this information has to be added to the note symbol.
                note_alteration = int(note.find("alter").string)
                if note_alteration == -1:
                    note_symbol += "b"
                elif note_alteration == 1:
                    note_symbol += "#"
            except AttributeError:
                pass
            final_note_symbol = f"{note_symbol}{note_octave}"
            note_to_add = Note(final_note_symbol, note_duration, offset)

        # If an AttributeError occurred, we know that the element
        #   is a rest. The duration and offset apply nonetheless.
        except AttributeError:
            note_to_add = RestNote(note_duration, offset)

        return note_to_add

    def _parse_one_harmony_symbol(
        self, harmony: bs4.element.Tag, offset: int
    ) -> Union[RestChord, Chord]:
        """Parse one harmony symbol and return it."""
        root = harmony.find("root-step").string
        chord_type = harmony.find("kind").get("text")

        new_offset = harmony.find("offset")
        if new_offset is not None:
            offset = new_offset

        if chord_type == "N.C.":
            chord_to_add = RestChord(offset=offset)
        else:
            if chord_type is None:
                chord_type = harmony.find("kind").string
            chord_type = self._convert_to_compatible_chord_type(chord_type)
            final_chord_symbol = f"{root}{chord_type}"
            chord_to_add = Chord(final_chord_symbol, offset=offset)
        return chord_to_add

    @staticmethod
    def _convert_to_compatible_chord_type(chord_type: str) -> str:
        return CHORD_TYPE_TO_COMPATIBLE_CHORD[chord_type]

    @staticmethod
    def _augment_training_data(
        musical_elements: Union[List[Chord], List[Note]]
    ) -> np.ndarray:
        """Augment the training data by transposing to each key.

        This means that the neural net will receive the training
        data in all keys to (1.) avoid over-representation of
        some key signatures (some key signatures are more common
        than others in jazz) and to (2.) lead to generalization
        (the network should understand that the relations between
        notes and chords are the same for all keys alike.
        """
        augmented_data = [
            element.transpose(transpose_steps).neural_net_representation
            for transpose_steps in range(-6, 6)
            for element in musical_elements
        ]
        return np.array(augmented_data)

    @property
    def melody_neural_net_representation(self) -> np.ndarray:
        """Represent the song as the input format the neural net.

        The notes are represented as a N*12 x 3 array where
        N is the number of notes in the song. The notes are
        transposed to all keys to augment the dataset.
        """
        note_heights = self._augment_training_data(self.melody_representation)
        note_durations = [
            note.duration for _ in range(12) for note in self.melody_representation
        ]
        note_offsets = [
            note.offset for _ in range(12) for note in self.melody_representation
        ]
        return np.vstack([note_heights, note_durations, note_offsets])

    @property
    def harmony_neural_net_representation(self) -> np.ndarray:
        """Represent the harmony of the song as the input format to the neural net."""
        chord_types = self._augment_training_data(self.harmony_representation)
        chord_offsets = np.array(
            [chord.offset for chord in self.harmony_representation]
        )
        chord_durations = self._calculate_chord_offsets(chord_offsets)
        chord_offsets = np.tile(chord_offsets, 12)
        chord_durations = np.tile(chord_durations, 12)
        return np.vstack([chord_types, chord_durations, chord_offsets])

    @staticmethod
    def _calculate_chord_offsets(chord_offsets: np.ndarray) -> List[int]:
        chord_offsets_offset_by_one = np.roll(chord_offsets, -1)
        chord_durations = chord_offsets_offset_by_one - chord_offsets
        chord_durations[chord_durations <= 0] += 48
        return chord_durations
