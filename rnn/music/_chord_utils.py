"""Utils for chords."""
import numpy as np

# Convert possible representations from .xml files into compatible ones.
_possible_minor_symbols = [
    "minor-seventh",
    "m",
    "m7",
    "m9",
    "m6",
    "m11",
    "m13",
    "m(#5)",
    "mi6",
    "mi9",
    "minor-sixth",
    "minor",
]

_possible_major_seven_symbols = [
    "major-seventh",
    "maj7",
    "mmaj7",
    "major",
    "maj9",
    "maj11",
    "maj13",
    "6",
    "",
    "ma7",
    "maj",
]

_possible_half_diminished_symbols = [
    "half-diminished",
    "o7",
    "dim7",
    "ø7",
    "diminished-seventh",
    "mi7b5",
    "m7b5",
    "dim",
    "diminished",
    "o",
    "ø",
]

_possible_dominant_seven_symbols = [
    "dominant-seventh",
    "sus4",
    "suspended-fourth",
    "7",
    "7#9",
    "13",
    "7b9",
    "7b5",
    "b9",
    "9",
    "+",
    "augmented",
]

_chord_type_to_compatible_chord = {}
_chord_type_to_compatible_chord.update(
    dict.fromkeys(_possible_major_seven_symbols, "maj7")
)
_chord_type_to_compatible_chord.update(dict.fromkeys(_possible_minor_symbols, "min7"))
_chord_type_to_compatible_chord.update(
    dict.fromkeys(_possible_dominant_seven_symbols, "dom7")
)
_chord_type_to_compatible_chord.update(
    dict.fromkeys(_possible_half_diminished_symbols, "dim7")
)

# Neural net representation of chords.
_keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_alternative_keys = [
    "B#",
    "Db",
    "D",
    "Eb",
    "Fb",
    "E#",
    "Gb",
    "G",
    "Ab",
    "A",
    "Bb",
    "Cb",
]
_chord_types = ["maj7", "min7", "dom7", "dim7"]

_chords = ["N.C."] + [f"{key} {chord}" for key in _keys for chord in _chord_types]
_alternative_chords = ["N.C."] + [
    f"{key} {chord}" for key in _alternative_keys for chord in _chord_types
]
_chords.append("N.C.")
_chord_symbol_to_neural_net_representation = {
    chord: i for i, chord in enumerate(_chords)
}
_chord_symbol_to_neural_net_representation.update(
    {chord: i for i, chord in enumerate(_alternative_chords)}
)

_neural_net_representation_to_chord_symbol = {
    value: key for key, value in _chord_symbol_to_neural_net_representation.items()
}

# Represent chord types as the function of the notes that the chord consists of.
_chord_types_to_numbers = {
    "min7": np.array([0, 3, 7, 10]),
    "maj7": np.array([0, 4, 7, 11]),
    "dim7": np.array([0, 3, 6, 10]),
    "dom7": np.array([0, 4, 7, 10]),
}
