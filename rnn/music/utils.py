"""Utils functions for the music package."""
from typing import Union

import numpy as np

SHARPS_TO_KEY_SIGNATURE_SYMBOL = {
    0: "C",
    -1: "F",
    -2: "Bb",
    -3: "Eb",
    -4: "Ab",
    -5: "Db",
    -6: "Gb",
    1: "G",
    2: "D",
    3: "A",
    4: "E",
    5: "B",
    6: "F#",
}

NOTE_SYMBOL_TO_NUMBER = {
    "C": 0,
    "Db": 1,
    "C#": 1,
    "D": 2,
    "Eb": 3,
    "D#": 3,
    "Fb": 4,
    "E": 4,
    "F": 5,
    "Gb": 6,
    "F#": 6,
    "G": 7,
    "Ab": 8,
    "G#": 8,
    "A": 9,
    "Bb": 10,
    "A#": 10,
    "B": 11,
}

MIDI_NUMBER_TO_NOTE_SYMBOL = {
    value: key for key, value in NOTE_SYMBOL_TO_NUMBER.items()
}

CHORD_TYPES_TO_NUMBERS = {
    "min7": np.array([0, 3, 7, 10]),
    "maj7": np.array([0, 4, 7, 11]),
    "dim7": np.array([0, 3, 6, 10]),
    "dom7": np.array([0, 4, 7, 10]),
}

CHORD_TYPE_TO_COMPATIBLE_CHORD = {}
CHORD_TYPE_TO_COMPATIBLE_CHORD.update(
    dict.fromkeys(["maj7", "major", "maj9", "maj11", "maj13", "6", ""], "maj7")
)
CHORD_TYPE_TO_COMPATIBLE_CHORD.update(
    dict.fromkeys(["7", "7#9", "13", "7b9", "7b5", "b9", "9", "+"], "dom7")
)
CHORD_TYPE_TO_COMPATIBLE_CHORD.update(
    dict.fromkeys(["half-diminished", "o7", "dim7", "dim", "ø7"], "dim7")
)
CHORD_TYPE_TO_COMPATIBLE_CHORD.update(
    dict.fromkeys(["m7", "m9", "m6", "m11", "m13", "m(#5)"], "min7")
)

keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
alternative_keys = ["B#", "Db", "D", "Eb", "Fb", "E#", "Gb", "G", "Ab", "A", "Bb", "Cb"]
chord_types = ["maj7", "min7", "dom7", "dim7"]

chords = ["N.C."] + [key + chord for key in keys for chord in chord_types]
chords_alternative_representation = ["N.C."] + [
    key + chord for key in alternative_keys for chord in chord_types
]
chords.append("N.C.")
chorddict = {chord: i for i, chord in enumerate(chords)}
chorddict.update(
    {chord: i for i, chord in enumerate(chords_alternative_representation)}
)
CHORD_TYPES_TO_NEURAL_NET_REPRESENTATION = chorddict


def functional_chord_notes_to_chord_symbol(chord_notes: np.ndarray) -> str:
    """Get the chord symbol based on the function that the notes have.

    For example, a maj7 chord is (in our framework) always constituted
    of the root, the major third, the fifth, and the raised seven.
    When counting these in half-tone steps, one can infer the chord
    type by looking at the functional relation of the chord notes
    in relation to the root note (the first number in the array).
    """
    if np.all(chord_notes == np.array([0, 3, 7, 10])):
        return "min7"
    if np.all(chord_notes == np.array([0, 4, 7, 11])):
        return "maj7"
    if np.all(chord_notes == np.array([0, 3, 6, 10])):
        return "dim7"
    if np.all(chord_notes == np.array([0, 4, 7, 10])):
        return "dom7"
    else:
        raise ValueError("The notes do not correspond to a chord type.")


def octave_in_range(octave: int) -> bool:
    """Indicate whether a given octave is within the plausible range."""
    return (octave >= 0) & (octave <= 8)


def pitch_height_in_range(pitch_height: Union[int, np.ndarray]) -> bool:
    """Indicate whether a given pitch height is within the plausible range.

    MIDI pitches range from 21 to 108, with 21 representing A0 and 108 C8;
    therefore, this is taken as the plausible range.
    """
    return np.all(pitch_height >= 21) & np.all(pitch_height <= 108)
