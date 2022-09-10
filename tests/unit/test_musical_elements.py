"""Test the musical elements."""
import numpy as np
import pytest

from rnn.music.musical_elements import Chord
from rnn.music.musical_elements import Note


class TestNote:
    """Test suite for the Note class."""

    @pytest.mark.parametrize("pitch", [60, 52, 46, 75])
    def test_instantiation_from_pitch_works(self, pitch):
        note = Note.from_pitch_height(pitch, 12, 0)
        assert note.pitch_height == pitch

    @pytest.mark.parametrize(
        "note_symbol, pitch_height",
        [("G#4", 68), ("G2", 43), ("F#6", 90), ("C3", 48), ("B5", 83)],
    )
    def test_instantiation_from_symbol_works(self, note_symbol, pitch_height):
        note = Note.from_symbol(note_symbol)
        assert note.symbol == note_symbol
        assert note.pitch_height == pitch_height

    @pytest.mark.parametrize("impossible_midi_value", [1, 12, -12, 409, 180, 109, 20])
    def test_instantiation_from_midi_values_works(self, impossible_midi_value):
        with pytest.raises(ValueError):
            Note.from_pitch_height(impossible_midi_value)

    @pytest.mark.parametrize(
        "note_symbol,transpose_steps,expected_resulting_symbol",
        [("G#4", 3, "B4"), ("C4", -2, "A#3"), ("D6", 12, "D7"), ("G3", 34, "F6")],
    )
    def test_note_symbol_transposing_works(
        self, note_symbol, expected_resulting_symbol, transpose_steps
    ):
        note = Note.from_symbol(note_symbol).transpose(transpose_steps)
        assert note.symbol == expected_resulting_symbol

    @pytest.mark.parametrize(
        "note_pitch, transpose_steps, expected_result",
        [(60, 11, 71), (72, 5, 77), (58, -5, 53), (42, -6, 36), (34, 0, 34)],
    )
    def test_pitch_heights_are_transposed_correctly(
        self, note_pitch, transpose_steps, expected_result
    ):
        note = Note(note_pitch)
        new_note = note.transpose(transpose_steps)
        assert new_note.pitch_height == expected_result

    @pytest.mark.parametrize(
        "note_pitch, note_duration, note_offset, transpose_steps",
        [(60, 12, 0, 5), (45, 24, 12, -4), (78, 12, 36, -6), (63, 48, 0, 6)],
    )
    def test_attributes_are_kept_when_transposing(
        self, note_pitch, note_duration, note_offset, transpose_steps
    ):
        note = Note(note_pitch, note_duration, note_offset)
        new_note = note.transpose(transpose_steps)
        assert new_note.duration == note.duration
        assert new_note.offset == note.offset

    @pytest.mark.parametrize(
        "note_symbol, midi_note", [("G#4", 68), ("C5", 72), ("D#4", 63), ("C1", 24)]
    )
    def test_pitch_heights_work(self, note_symbol, midi_note):
        """Test if the pitch heights work."""
        note = Note.from_symbol(note_symbol)
        assert note.pitch_height == midi_note

    @pytest.mark.parametrize(
        "note_symbol, octave", [("Gb3", 3), ("F#5", 5), ("B3", 3), ("D2", 2)]
    )
    def test_octaves_work(self, note_symbol, octave):
        assert Note.from_symbol(note_symbol).octave == octave

    @pytest.mark.parametrize(
        "note_symbol,transpose_steps,expected_result",
        [("G#4", 11, 79), ("C5", -2, 70), ("D#4", -4, 59), ("C1", 24, 48)],
    )
    def test_transposing_pitch_heights_work(
        self, note_symbol, expected_result, transpose_steps
    ):
        """Test if the pitch heights work."""
        note = Note.from_symbol(note_symbol).transpose(transpose_steps)
        assert np.all(note.pitch_height == expected_result)

    def test_transposing_octaves_up_and_down_works(self):
        note = Note.from_symbol("G4")
        assert note.octave_up().symbol == "G5"
        assert note.octave_up().pitch_height == 79

    @pytest.mark.parametrize(
        "note1, note2, expected_result",
        [
            (Note.from_symbol("G3", 12, 0), Note.from_symbol("G3", 12, 12), False),
            (Note.from_symbol("D#3", 12, 0), Note.from_symbol("D#4", 12, 0), False),
            (Note.from_symbol("Ab4", 24, 0), Note.from_symbol("G#4", 24, 0), True),
            (Note.from_symbol("F3", 12, 36), Note.from_symbol("F3", 12, 36), True),
            (Note.from_pitch_height(53, 12, 36), Note.from_symbol("F3", 12, 36), True),
            (Note(60, 12, 0), Note(60, 12, 12), False),
            (
                Note.from_neural_net_representation(12, 12, 0),
                Note(12 + 48, 12, 0),
                True,
            ),
        ],
    )
    def test_equality_method_works(self, note1, note2, expected_result):
        comparison_result = note1 == note2
        assert comparison_result == expected_result


class TestChord:
    """Test suite for the Chord class."""

    @pytest.mark.parametrize(
        "chord_symbol, expected_notes",
        [
            (
                "G# min7",
                [
                    Note.from_symbol("G#4"),
                    Note.from_symbol("B4"),
                    Note.from_symbol("D#5"),
                    Note.from_symbol("F#5"),
                ],
            ),
            (
                "D dom7",
                [
                    Note.from_symbol("D4"),
                    Note.from_symbol("F#4"),
                    Note.from_symbol("A4"),
                    Note.from_symbol("C5"),
                ],
            ),
            (
                "C min7",
                [
                    Note.from_symbol("C4"),
                    Note.from_symbol("D#4"),
                    Note.from_symbol("G4"),
                    Note.from_symbol("A#4"),
                ],
            ),
            (
                "F dim7",
                [
                    Note.from_symbol("F4"),
                    Note.from_symbol("G#4"),
                    Note.from_symbol("B4"),
                    Note.from_symbol("D#5"),
                ],
            ),
            (
                "A# dom7",
                [
                    Note.from_symbol("A#4"),
                    Note.from_symbol("D5"),
                    Note.from_symbol("F5"),
                    Note.from_symbol("G#5"),
                ],
            ),
            (
                "B maj7",
                [
                    Note.from_symbol("B4"),
                    Note.from_symbol("D#5"),
                    Note.from_symbol("F#5"),
                    Note.from_symbol("A#5"),
                ],
            ),
        ],
    )
    def test_instantiation_from_symbol_works(self, chord_symbol, expected_notes):
        """Test if the chord symbols work."""
        chord = Chord.from_symbol(chord_symbol)
        assert chord.symbol == chord_symbol
        assert chord.notes == expected_notes

    @pytest.mark.parametrize(
        "pitch_height, expected_notes",
        [
            (
                np.r_[68, 71, 75, 78],
                [
                    Note.from_symbol("G#4"),
                    Note.from_symbol("B4"),
                    Note.from_symbol("D#5"),
                    Note.from_symbol("F#5"),
                ],
            ),
            (
                np.r_[62, 66, 69, 72],
                [
                    Note.from_symbol("D4"),
                    Note.from_symbol("F#4"),
                    Note.from_symbol("A4"),
                    Note.from_symbol("C5"),
                ],
            ),
            (
                np.r_[60, 63, 67, 70],
                [
                    Note.from_symbol("C4"),
                    Note.from_symbol("D#4"),
                    Note.from_symbol("G4"),
                    Note.from_symbol("A#4"),
                ],
            ),
            (
                np.r_[65, 68, 71, 75],
                [
                    Note.from_symbol("F4"),
                    Note.from_symbol("G#4"),
                    Note.from_symbol("B4"),
                    Note.from_symbol("D#5"),
                ],
            ),
            (
                np.r_[70, 74, 77, 80],
                [
                    Note.from_symbol("A#4"),
                    Note.from_symbol("D5"),
                    Note.from_symbol("F5"),
                    Note.from_symbol("G#5"),
                ],
            ),
            (
                np.r_[71, 75, 78, 82],
                [
                    Note.from_symbol("B4"),
                    Note.from_symbol("D#5"),
                    Note.from_symbol("F#5"),
                    Note.from_symbol("A#5"),
                ],
            ),
        ],
    )
    def test_instantiation_from_pitch_height_works(self, pitch_height, expected_notes):
        """Test if the instantiation from pitch heights works."""
        chord = Chord.from_pitch_height(pitch_height)
        assert np.all(chord.pitch_height == pitch_height)
        assert chord.notes == expected_notes

    @pytest.mark.parametrize(
        "chord_symbol, transpose_steps, expected_notes",
        [
            (
                "G# min7",
                4,
                [
                    Note.from_symbol("G#4").transpose(4),
                    Note.from_symbol("B4").transpose(4),
                    Note.from_symbol("D#5").transpose(4),
                    Note.from_symbol("F#5").transpose(4),
                ],
            ),
            (
                "D dom7",
                10,
                [
                    Note.from_symbol("D4").transpose(10),
                    Note.from_symbol("F#4").transpose(10),
                    Note.from_symbol("A4").transpose(10),
                    Note.from_symbol("C5").transpose(10),
                ],
            ),
            (
                "C min7",
                -2,
                [
                    Note.from_symbol("C4").transpose(-2),
                    Note.from_symbol("D#4").transpose(-2),
                    Note.from_symbol("G4").transpose(-2),
                    Note.from_symbol("A#4").transpose(-2),
                ],
            ),
            (
                "F dim7",
                -5,
                [
                    Note.from_symbol("F4").transpose(-5),
                    Note.from_symbol("G#4").transpose(-5),
                    Note.from_symbol("B4").transpose(-5),
                    Note.from_symbol("D#5").transpose(-5),
                ],
            ),
            (
                "A# dom7",
                -12,
                [
                    Note.from_symbol("A#4").transpose(-12),
                    Note.from_symbol("D5").transpose(-12),
                    Note.from_symbol("F5").transpose(-12),
                    Note.from_symbol("G#5").transpose(-12),
                ],
            ),
            (
                "B maj7",
                -2,
                [
                    Note.from_symbol("B4").transpose(-2),
                    Note.from_symbol("D#5").transpose(-2),
                    Note.from_symbol("F#5").transpose(-2),
                    Note.from_symbol("A#5").transpose(-2),
                ],
            ),
        ],
    )
    def test_transposing_works(self, chord_symbol, transpose_steps, expected_notes):
        chord = Chord.from_symbol(chord_symbol).transpose(transpose_steps)
        assert chord.notes == expected_notes

    @pytest.mark.parametrize(
        "chord1, chord2, expected_result",
        [
            (
                Chord.from_symbol("G maj7", 12, 0),
                Chord.from_symbol("G maj7", 12, 12),
                False,
            ),
            (
                Chord.from_symbol("D# dim7", 12, 0),
                Chord.from_symbol("D# dom7", 12, 0),
                False,
            ),
            (
                Chord.from_symbol("Ab maj7", 24, 0),
                Chord.from_symbol("G# maj7", 24, 0),
                True,
            ),
            (
                Chord.from_symbol("C min7", 12, 36),
                Chord.from_pitch_height(np.r_[60, 63, 67, 70], 12, 36),
                True,
            ),
            (
                Chord.from_symbol("C min7", 12, 12),
                Chord(
                    [
                        Note(60, 12, 24),
                        Note(63, 12, 12),
                        Note(67, 24, 30),
                        Note(70, 12, 12),
                    ],
                    duration=12,
                    offset=12,
                ),
                True,
            ),
        ],
    )
    def test_equality_method_works(self, chord1, chord2, expected_result):
        comparison_result = chord1 == chord2
        assert comparison_result == expected_result
