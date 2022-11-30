"""An input creator for the neural net predictions."""
import glob
import random

import numpy as np

from rnn.music import Note
from rnn.music.song import HarmonySongParser
from rnn.music.song import MelodySongParser
from rnn.music.song import SongParser


class InputCreator(SongParser):
    """Create an input for the neural net."""

    def __init__(
        self,
        key: str = "C major",
        input_length: int = 8,
        last_measures: bool = True,
        file: str = None,
    ) -> None:
        """Initialize the input creator."""
        if file is None:
            files = glob.glob("../../data/*.xml")
            file = random.choice(files)
        super().__init__(file)
        self.key = key
        self.input_length = input_length
        self.last_measures = last_measures

    def _find_transposing_steps(self):
        """Find the number of transposing steps necessary to get to the desired key."""
        major = self.key_signature[-1] != "m"
        key_symbol = self.key_signature[0] if major else self.key_signature[0][:-1]
        current_key = Note.from_symbol(f"{key_symbol}4").pitch_height
        target_key = Note.from_symbol(f"{self.key.split()[0]}4").pitch_height
        transposing_steps = target_key - current_key
        return transposing_steps


class MelodyInputCreator(InputCreator, MelodySongParser):
    """Create a melody input for the neural net."""

    def get_neural_net_input(self):
        """Get the input for the neural net."""
        self.parse()
        transposing_steps = self._find_transposing_steps()
        if self.last_measures:
            relevant_melody_part = self.melody_representation[-self.input_length :]
            relevant_chord_part = self.harmony_representation[-self.input_length :]
        else:
            relevant_melody_part = self.melody_representation[: self.input_length]
            relevant_chord_part = self.harmony_representation[: self.input_length]
        transposed_notes = [
            note.transpose(transposing_steps) for note in relevant_melody_part
        ]
        transposed_chords = [
            chord.transpose(transposing_steps) for chord in relevant_chord_part
        ]
        notes = [note.neural_net_representation for note in transposed_notes]
        durations = [note.duration for note in transposed_notes]
        offsets = [note.offset for note in transposed_notes]
        chords = [
            chord.transpose(transposing_steps).pitch_neural_net_representation
            for chord in transposed_chords
        ]
        chord_note_1, chord_note_2, chord_note_3, chord_note_4 = chords
        return np.vstack(
            [
                np.array(
                    [
                        notes,
                        durations,
                        offsets,
                        chord_note_1,
                        chord_note_2,
                        chord_note_3,
                        chord_note_4,
                    ]
                )
            ]
        )
        # return np.vstack([np.array([notes, durations, offsets]), np.array(chords).T])


class HarmonyInputCreator(InputCreator, HarmonySongParser):
    """Create a harmony input for the neural net."""

    def get_neural_net_input(self):
        """Get the input for the neural net."""
        self.parse()
        transposing_steps = self._find_transposing_steps()
        if self.last_measures:
            relevant_chord_part = self.harmony_representation[-self.input_length :]
        else:
            relevant_chord_part = self.harmony_representation[: self.input_length]
        transposed_chords = [
            chord.transpose(transposing_steps) for chord in relevant_chord_part
        ]
        chords = [chord.neural_net_representation for chord in transposed_chords]
        durations = [chord.duration for chord in transposed_chords]
        offsets = [chord.offset for chord in transposed_chords]
        return np.array([chords, durations, offsets])
