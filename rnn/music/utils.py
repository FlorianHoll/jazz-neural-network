"""Utils functions for the music package."""

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


def octave_in_range(octave: int) -> bool:
    """Indicate whether a given octave is within the plausible range."""
    return (octave >= 0) & (octave <= 8)


def pitch_height_in_range(pitch_height: int) -> bool:
    """Indicate whether a given pitch height is within the plausible range.

    MIDI pitches range from 21 to 108, with 21 representing A0 and 108 C8;
    therefore, this is taken as the plausible range.
    """
    return (pitch_height >= 21) & (pitch_height <= 108)
