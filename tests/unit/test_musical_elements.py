"""Test the musical elements."""
import numpy as np
import pytest

from rnn.music.musical_elements import Chord
from rnn.music.musical_elements import MidiChord
from rnn.music.musical_elements import MidiNote
from rnn.music.musical_elements import Note


class TestNote:
    """Test suite for the Note class."""

    @pytest.mark.parametrize("note_symbol", ["G#4", "G2", "Ab6", "C3", "Db3"])
    def test_note_symbol_works(self, note_symbol):

        note = Note(note_symbol)
        assert note.symbol == note_symbol

    @pytest.mark.parametrize(
        "note_symbol,transpose_steps,expected_resulting_symbol",
        [("G#4", 3, "B4"), ("C4", -2, "A#3"), ("D6", 7, "A6"), ("G3", 34, "F6")],
    )
    def test_note_symbol_transposing_works(
        self, note_symbol, expected_resulting_symbol, transpose_steps
    ):
        note = Note(note_symbol).transpose(transpose_steps)
        assert note.symbol == expected_resulting_symbol

    @pytest.mark.parametrize(
        "note_symbol,midi_note", [("G#4", 68), ("C5", 72), ("D#4", 63), ("C1", 24)]
    )
    def test_pitch_heights_work(self, note_symbol, midi_note):
        """Test if the pitch heights work."""
        note = Note(note_symbol)
        assert note.pitch_height == midi_note

    @pytest.mark.parametrize(
        "note_symbol,transpose_steps,expected_result",
        [("G#4", 11, 79), ("C5", -2, 70), ("D#4", -4, 59), ("C1", 24, 48)],
    )
    def test_transposing_pitch_heights_work(
        self, note_symbol, expected_result, transpose_steps
    ):
        """Test if the pitch heights work."""
        note = Note(note_symbol).transpose(transpose_steps)
        assert np.all(note.pitch_height == expected_result)
        assert np.all(note.octave_up().pitch_height == expected_result + 12)
        assert np.all(note.octave_down().pitch_height == expected_result - 12)


class TestMidiNote:
    """Test suite for the MidiNote class."""

    @pytest.mark.parametrize("note_height", [21, 49, 100, 48, 68])
    def test_note_symbol_works(self, note_height):
        note = MidiNote(note_height)
        assert note.pitch_height == note_height

    @pytest.mark.parametrize(
        "note_height,expected_resulting_symbol",
        [(70, "A#4"), (100, "E7"), (27, "D#1"), (58, "A#3")],
    )
    def test_note_height_works(
        self,
        note_height,
        expected_resulting_symbol,
    ):
        note = MidiNote(note_height)
        assert note.symbol == expected_resulting_symbol

    @pytest.mark.parametrize(
        "note_height,transpose_steps,expected_resulting_symbol",
        [(70, 2, "C5"), (100, 7, "B7"), (27, -4, "B0"), (58, 0, "A#3")],
    )
    def test_note_height_transposing_works(
        self, note_height, expected_resulting_symbol, transpose_steps
    ):
        """Test if the note symbols work as expected when transposing."""
        note = MidiNote(note_height).transpose(transpose_steps)
        assert note.symbol == expected_resulting_symbol

    @pytest.mark.parametrize("note_height", [0, 20, 109, 1293, 815, -123, -1, -50])
    def test_instantiation_fails_for_pitches_outside_range(self, note_height):
        with pytest.raises(ValueError):
            MidiNote(note_height)


class TestChord:
    """Test suite for the Chord class."""

    def test_chord_symbol_works(self):
        """Test if the chord symbols work."""
        chord_symbol = "Abmin7"
        chord = Chord(chord_symbol)
        assert chord.symbol == chord_symbol

    @pytest.mark.parametrize(
        "chord_symbol,transpose_steps,expected_resulting_symbol",
        [
            ("G#dim7", 3, "Bdim7"),
            ("Cmin7", -2, "A#min7"),
            ("Dmaj7", 7, "Amaj7"),
            ("Abdim7", -2, "F#dim7"),
        ],
    )
    def test_chord_symbol_transposing_works(
        self, chord_symbol, expected_resulting_symbol, transpose_steps
    ):
        """Test if the chord symbols work as expected when transposing."""
        chord = Chord(chord_symbol).transpose(transpose_steps)
        assert chord.symbol == expected_resulting_symbol

    @pytest.mark.parametrize(
        "chord_symbol,midi_notes",
        [
            ("G#dim7", np.r_[68, 71, 74, 78]),
            ("Cmin7", np.r_[60, 63, 67, 70]),
            ("Dmaj7", np.r_[62, 66, 69, 73]),
            ("Gdom7", np.r_[67, 71, 74, 77]),
        ],
    )
    def test_pitch_heights_work(self, chord_symbol, midi_notes):
        """Test if the pitch heights work."""
        chord = Chord(chord_symbol)
        assert np.all(chord.pitch_height == midi_notes)

    @pytest.mark.parametrize(
        "chord_symbol,transpose_steps,midi_notes",
        [
            ("G#dim7", 2, np.r_[70, 73, 76, 80]),
            ("Cmin7", -5, np.r_[55, 58, 62, 65]),
            ("Dmaj7", -1, np.r_[61, 65, 68, 72]),
            ("Gdom7", 3, np.r_[70, 74, 77, 80]),
        ],
    )
    def test_transposing_pitch_heights_work(
        self, chord_symbol, midi_notes, transpose_steps
    ):
        """Test if the pitch heights work."""
        chord = Chord(chord_symbol).transpose(transpose_steps)
        assert np.all(chord.pitch_height == midi_notes)
        assert np.all(chord.octave_up().pitch_height == midi_notes + 12)
        assert np.all(chord.octave_down().pitch_height == midi_notes - 12)


class TestMidiChord:
    """Test suite for the MidiChord class."""

    @pytest.mark.parametrize(
        "chord_pitches,expected_resulting_symbol",
        [
            (np.r_[60, 64, 67, 70], "Cdom7"),
            (np.r_[23, 26, 30, 33], "Bmin7"),
            (np.r_[61, 64, 67, 71], "C#dim7"),
            (np.r_[55, 59, 62, 66], "Gmaj7"),
        ],
    )
    def test_midi_chord_symbol_works(self, chord_pitches, expected_resulting_symbol):
        chord = MidiChord(chord_pitches)
        assert chord.symbol == expected_resulting_symbol

    @pytest.mark.parametrize(
        "chord_pitches,transpose_steps,expected_resulting_symbol",
        [
            (np.r_[60, 64, 67, 70], 2, "Ddom7"),
            (np.r_[23, 26, 30, 33], 4, "D#min7"),
            (np.r_[61, 64, 67, 71], 5, "F#dim7"),
            (np.r_[55, 59, 62, 66], -14, "Fmaj7"),
        ],
    )
    def test_midi_chord_symbol_works(
        self, chord_pitches, transpose_steps, expected_resulting_symbol
    ):
        """Test if the chord symbols work."""
        chord = MidiChord(chord_pitches).transpose(transpose_steps)
        assert np.all(chord.pitch_height == chord_pitches + transpose_steps)
        assert chord.symbol == expected_resulting_symbol

    @pytest.mark.parametrize(
        "chord_pitches",
        [
            (np.r_[-12, 312, 12, 12]),
            (np.r_[23, 26, 30, -12]),
            (np.r_[61, 64, 67, 721]),
            (np.r_[55, 59, 62, 109]),
        ],
    )
    def test_instantiation_fails_for_pitches_outside_range(self, chord_pitches):
        with pytest.raises(ValueError):
            MidiChord(chord_pitches)

    @pytest.mark.parametrize(
        "chord_pitches",
        [
            (np.r_[60, 64, 67, 72]),
            (np.r_[23, 26, 30, 47]),
            (np.r_[61, 64, 67, 21]),
            (np.r_[55, 59, 62, 64]),
        ],
    )
    def test_symbol_works_only_for_valid_chord_types(self, chord_pitches):
        chord = MidiChord(chord_pitches)
        with pytest.raises(ValueError):
            chord.symbol
