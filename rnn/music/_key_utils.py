"""Key utils."""

_sharps_to_key_signature_symbol = {
    0: ("C", "Am"),
    -1: ("F", "Dm"),
    -2: ("Bb", "Gm"),
    -3: ("Eb", "Cm"),
    -4: ("Ab", "Fm"),
    -5: ("Db", "Bbm"),
    -6: ("Gb", "Ebm"),
    1: ("G", "Em"),
    2: ("D", "Bm"),
    3: ("A", "F#m"),
    4: ("E", "C#m"),
    5: ("B", "G#m"),
    6: ("F#", "D#m"),
}

# Major keys
_key_to_sharps = {
    value[0]: key for key, value in _sharps_to_key_signature_symbol.items()
}

# Minor keys
_key_to_sharps.update(
    {value[1]: key for key, value in _sharps_to_key_signature_symbol.items()}
)
