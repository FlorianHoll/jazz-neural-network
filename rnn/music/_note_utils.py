"""Note utils."""

_note_symbol_to_number = {
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
    "Cb": 11,
    "B": 11,
}

_midi_number_to_note_symbol = {
    value: key for key, value in _note_symbol_to_number.items()
}

_NEURAL_NET_TRANSFORM_SCALING = 48
