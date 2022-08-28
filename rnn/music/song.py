"""A song in the training corpus."""
import numpy as np
from bs4 import BeautifulSoup

from rnn.music.musical_elements import Note
from rnn.music.musical_elements import RestNote
from rnn.music.utils import SHARPS_TO_KEY_SIGNATURE_SYMBOL


class Song:
    """A song that will be used to train the neural network."""

    def __init__(self, filename: str):
        """Initialize the song."""
        with open(filename, "r") as file:
            data = file.read()
        self.raw_data = BeautifulSoup(data, "xml")

        self.music_representation = []

    @property
    def key_signature(self):
        """Get the key signature of the whole song."""
        key_signature_in_fifths_from_c = int(
            self.raw_data.find("key").find("fifths").string
        )
        return SHARPS_TO_KEY_SIGNATURE_SYMBOL[key_signature_in_fifths_from_c]

    def _parse_one_measure(self, measure):
        notes = measure.findChildren("note")  # find all notes.
        # The offset is set to 0 initially and will be updated iteratively.
        offset = 0
        for note in notes:
            # The durations as written in the .xml have to be multiplied
            #   by four to represent our time grid.
            note_duration = int(note.find("duration").string) * 4
            # If the note is NOT a rest, it will have the 'step'
            #   and 'octave' attributes; however, if it IS a rest,
            #   an AttributeError will be raised.
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

            # Add the note to the music representation.
            self.music_representation.append(note_to_add)

            # Update the offset.
            offset += note_duration

    def parse(self):
        """Parse the song from the xml representation."""
        measures = self.raw_data.findChildren("measure")
        for measure in measures:
            self._parse_one_measure(measure)
