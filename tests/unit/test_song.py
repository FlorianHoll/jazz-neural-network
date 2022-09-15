"""Test the song."""
import bs4
import numpy as np
import pytest

from rnn.music.musical_elements import Chord
from rnn.music.song import HarmonySongParser
from rnn.music.song import SongParser


class TestSongParser:
    """Test suite for the Song base class."""

    @pytest.fixture
    def file_path(self):
        return "../test_data/test_song.xml"

    @pytest.fixture
    def wrong_fileformat_file_path(self):
        return "../test_data/test_song.mxl"

    def test_xml_file_can_be_loaded(self, file_path):
        song = SongParser(file_path)
        assert isinstance(song.raw_data, bs4.BeautifulSoup)

    def test_wrong_file_format_cannot_be_loaded(self, wrong_fileformat_file_path):
        with pytest.raises(TypeError):
            SongParser(wrong_fileformat_file_path)

    def test_key_signature_is_recognized(self, file_path):
        song = SongParser(file_path)
        assert song.key_signature == ("Eb", "Cm")


class TestHarmonySongParser:
    """Test suite for the Harmony Songparser class."""

    @pytest.fixture
    def file_path(self):
        return "../test_data/test_song.xml"

    @pytest.fixture
    def expected_harmony(self):
        expected_harmony = [
            Chord.from_symbol("C min7", 48, 0),
            Chord.from_symbol("C min7", 24, 0),
            Chord.from_symbol("F min7", 12, 24),
            Chord.from_symbol("Bb dom7", 12, 36),
            Chord.from_symbol("D dim7", 24, 0),
            Chord.from_symbol("G dom7", 24, 24),
            Chord.from_symbol("C min7", 24, 0),
            Chord.from_symbol("D dim7", 12, 24),
            Chord.from_symbol("G dom7", 12, 36),
        ]
        return expected_harmony

    def test_parsed_file_matches_expectations(self, file_path, expected_harmony):
        """
        Test whether the parsed song matches the expectation, i.e. the input.

        Since the song was only "written" for test purposes and is very short,
        we can simply hardcore the expected outcome.
        """
        song_parser = HarmonySongParser(file_path)
        song_parser.parse()
        song_parser._calculate_chord_durations()
        assert song_parser.harmony_representation == expected_harmony

    @pytest.mark.parametrize(
        "sequence, sequence_length, target_length, expected_result",
        [
            (
                np.arange(5),
                3,
                1,
                np.array(
                    [
                        [0, 1, 2, 3],
                        [1, 2, 3, 4],
                        [2, 3, 4, 0],
                        [3, 4, 0, 1],
                        [4, 0, 1, 2],
                    ]
                ),
            ),
            (
                np.arange(6),
                5,
                1,
                np.array(
                    [
                        [0, 1, 2, 3, 4, 5],
                        [1, 2, 3, 4, 5, 0],
                        [2, 3, 4, 5, 0, 1],
                        [3, 4, 5, 0, 1, 2],
                        [4, 5, 0, 1, 2, 3],
                        [5, 0, 1, 2, 3, 4],
                    ]
                ),
            ),
            (np.arange(3), 1, 1, np.array([[0, 1], [1, 2], [2, 0]])),
        ],
    )
    def test_sliding_window_works_as_expected(
        self, sequence, target_length, sequence_length, expected_result, file_path
    ):
        """Test whether the sliding window approach works."""
        result = HarmonySongParser._sliding_window(
            sequence, sequence_length, target_length
        )
        np.testing.assert_array_equal(result, expected_result)

    @pytest.mark.parametrize(
        "original_sequence, attribute, sequence_length, target_length, expected_result",
        [
            (
                [
                    Chord.from_symbol(symbol)
                    for symbol in ["C min7", "D dim7", "G dom7"]
                ],
                "neural_net_representation",
                2,
                1,
                np.array(
                    [
                        Chord.from_symbol(symbol).neural_net_representation
                        for symbol in np.array(
                            [
                                ["F# min7", "G# dim7", "C# dom7"],
                                ["G# dim7", "C# dom7", "F# min7"],
                                ["C# dom7", "F# min7", "G# dim7"],
                                ["G min7", "A dim7", "D dom7"],
                                ["A dim7", "D dom7", "G min7"],
                                ["D dom7", "G min7", "A dim7"],
                                ["G# min7", "A# dim7", "D# dom7"],
                                ["A# dim7", "D# dom7", "G# min7"],
                                ["D# dom7", "G# min7", "A# dim7"],
                                ["A min7", "B dim7", "E dom7"],
                                ["B dim7", "E dom7", "A min7"],
                                ["E dom7", "A min7", "B dim7"],
                                ["A# min7", "C dim7", "F dom7"],
                                ["C dim7", "F dom7", "A# min7"],
                                ["F dom7", "A# min7", "C dim7"],
                                ["B min7", "C# dim7", "F# dom7"],
                                ["C# dim7", "F# dom7", "B min7"],
                                ["F# dom7", "B min7", "C# dim7"],
                                ["C min7", "D dim7", "G dom7"],
                                ["D dim7", "G dom7", "C min7"],
                                ["G dom7", "C min7", "D dim7"],
                                ["C# min7", "D# dim7", "G# dom7"],
                                ["D# dim7", "G# dom7", "C# min7"],
                                ["G# dom7", "C# min7", "D# dim7"],
                                ["D min7", "E dim7", "A dom7"],
                                ["E dim7", "A dom7", "D min7"],
                                ["A dom7", "D min7", "E dim7"],
                                ["D# min7", "F dim7", "A# dom7"],
                                ["F dim7", "A# dom7", "D# min7"],
                                ["A# dom7", "D# min7", "F dim7"],
                                ["E min7", "F# dim7", "B dom7"],
                                ["F# dim7", "B dom7", "E min7"],
                                ["B dom7", "E min7", "F# dim7"],
                                ["F min7", "G dim7", "C dom7"],
                                ["G dim7", "C dom7", "F min7"],
                                ["C dom7", "F min7", "G dim7"],
                            ]
                        ).flatten()
                    ]
                ).reshape(36, 3),
            ),
            (
                [Chord.from_symbol("D maj7", duration) for duration in [6, 12, 24]],
                "duration",
                2,
                1,
                np.tile(np.array([[6, 12, 24], [12, 24, 6], [24, 6, 12]]), 12).T,
            ),
            (
                [Chord.from_symbol("D maj7", offset=offset) for offset in [0, 12, 36]],
                "offset",
                2,
                1,
                np.tile(np.array([[0, 12, 36], [12, 36, 0], [36, 0, 12]]), 12).T,
            ),
        ],
    )
    def test_transpose_and_sliding_window_works(
        self,
        file_path,
        original_sequence,
        sequence_length,
        target_length,
        expected_result,
        attribute,
    ):
        song_parser = HarmonySongParser(file_path)
        result = song_parser._transpose_and_slide(
            original_sequence,
            attribute,
            input_length=sequence_length,
            target_length=target_length,
        )
        assert np.all(result == expected_result)
