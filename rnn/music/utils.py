"""Utils functions for the music package."""
from typing import Union

import numpy as np

from rnn.music._chord_utils import _chord_symbol_to_neural_net_representation
from rnn.music._chord_utils import _chord_type_to_compatible_chord
from rnn.music._chord_utils import _chord_types_to_numbers
from rnn.music._chord_utils import _neural_net_representation_to_chord_symbol
from rnn.music._key_utils import _key_to_sharps
from rnn.music._key_utils import _sharps_to_key_signature_symbol
from rnn.music._note_utils import _midi_number_to_note_symbol
from rnn.music._note_utils import _NEURAL_NET_TRANSFORM_SCALING
from rnn.music._note_utils import _note_symbol_to_number


def key_symbol_to_sharps(key: str) -> int:
    """Convert a key symbol to the corresponding amount of sharps or flats.

    Positive values correspond to sharps, negative ones to flats. C major is
    therefore 0, Eb major -3, D major 2.
    This is needed for the prediction to convert a song into the desired key.

    :param key: The key to convert.
    :return: The sharps or flats of the key.
    """
    return _key_to_sharps[key]


def sharps_to_key_signature_symbol(number_sharps: int) -> tuple[str]:
    """Convert the number of sharps to a key signature.

    :param number_sharps: The number of sharps, where a positive
        number indicates the number of sharps and a negative number
        indicates the number of flats. For example, 0 is (C, Am),
        4 would mean 4 sharps, i.e. (E, C#m), -3 would mean 3 flats,
        i.e. (Eb, Cm).
    :return: The key signature as a tuple of both the major and the
        minor key signature with the corresponding number of flats or
        sharps, e.g. (C, Am).
    """
    return _sharps_to_key_signature_symbol[number_sharps]


def note_symbol_to_number(note_symbol: str) -> int:
    """Convert a note symbol to the number in the octave.

    :param note_symbol: The note symbol (e.g. "G", "Bb").
    :return: The number of the note in the octave (e.g. 7, 10).
    """
    return _note_symbol_to_number[note_symbol]


def midi_number_to_note_symbol(midi_number: int) -> str:
    """Convert a midi number to a note symbol.

    :param midi_number: The midi number (e.g. 60).
    :return: The corresponding symbol ("C").
    """
    return _midi_number_to_note_symbol[midi_number]


def chord_symbol_to_neural_net_representation(chord_type: str) -> int:
    """Convert a chord symbol to the neural network representation.

    :param chord_type: The chord symbol (e.g. "C#min7" or "Fmaj7").
    :return: The neural net representation of the chord symbol (e.g. 37, 29).
    """
    return _chord_symbol_to_neural_net_representation[chord_type]


def neural_net_representation_to_chord_symbol(neural_net_chord_number: int) -> str:
    """Convert a chord symbol to the neural network representation.

    :param neural_net_chord_number: The neural net representation
        of the chord symbol (e.g. 37, 29).
    :return: The chord symbol (e.g. "C#min7" or "Fmaj7").
    """
    return _neural_net_representation_to_chord_symbol[neural_net_chord_number]


def functional_chord_notes_to_chord_symbol(chord_notes: np.ndarray) -> str:
    """Get the chord symbol based on the function that the notes have.

    For example, a maj7 chord is (in our framework) always constituted
    of the root, the major third, the fifth, and the raised seven.
    When counting these in half-tone steps, one can infer the chord
    type by looking at the functional relation of the chord notes
    in relation to the root note (the first number in the array).
    """
    for key, value in _chord_types_to_numbers.items():
        if np.all(chord_notes == value):
            return key
    raise ValueError("The notes do not correspond to a chord type.")


def chord_type_to_numbers(chord_type: str) -> np.ndarray:
    """Convert a chord type to the functional numbers of its notes.

    :param chord_type: The chord type, e.g. "min7", "maj7".
    :return: The notes that the chord consists of.
    """
    return _chord_types_to_numbers[chord_type]


def chord_type_to_compatible_chord(chord_type: str) -> str:
    """Convert any .xml representation of chord type into a compatible one.

    :param chord_type: The chord type from the .xml file.
    :return: The chord type that is compatible with the MusicalElement classes.
    """
    return _chord_type_to_compatible_chord[chord_type]


def octave_in_range(octave: int) -> bool:
    """Indicate whether a given octave is within the plausible range."""
    return (octave >= 0) & (octave <= 8)


def pitch_heights_in_range(pitch_height: Union[int, np.ndarray]) -> bool:
    """Indicate whether a given pitch height is within the plausible range.

    MIDI pitches range from 21 to 108, with 21 representing A0 and 108 C8;
    therefore, this is taken as the plausible range.

    :param pitch_height: The MIDI pitch height(s) (either a single note or a chord).
    :return: Boolean indicating if all given pitch heights are in the plausible range.
    """
    return np.all(pitch_height >= 21) & np.all(pitch_height <= 108)


def note_to_neural_net_representation(pitch_height: Union[int, np.ndarray]) -> int:
    """Transform a note into the neural net representation.

    :param pitch_height: The MIDI pitch height(s) (either a single note or a chord).
    :return: The neural net representation of the note.
    """
    return pitch_height - _NEURAL_NET_TRANSFORM_SCALING


def neural_net_representation_to_note(pitch_height: Union[int, np.ndarray]) -> int:
    """Transform a note into the neural net representation.

    :param pitch_height: The neural net representation(s) (either a single note
        or a chord).
    :return: The MIDI representation of the note(s).
    """
    return pitch_height + _NEURAL_NET_TRANSFORM_SCALING
