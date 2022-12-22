"""A pipeline for predicting, i.e. for writing a song."""
from typing import List

import numpy as np

from rnn.model.chord_model import ChordModel
from rnn.model.melody_model import MelodyModel
from rnn.music import Chord
from rnn.music.song import HarmonySongParser
from rnn.prediction.greedy_search import GreedySearch
from rnn.prediction.greedy_search import MelodyGreedySearch
from rnn.prediction.input_creator import HarmonyInputCreator
from rnn.prediction.input_creator import MelodyInputCreator


class SongPredictor:
    """Pipeline for predicting a song from the trained models."""

    DEFAULT_WEIGHT_LOCATION = "../training/model/trained_models"

    def __init__(
        self,
        key: str,
        chord_model_weights_location: str = None,
        melody_model_weights_location: str = None,
        nr_measures: int = 32,
        filename: str = None,
        harmony: List[Chord] = None,
    ) -> None:
        """Initialize the prediction pipeline."""
        self.key = key
        self.nr_measures = nr_measures
        if not chord_model_weights_location:
            chord_model_weights_location = f"{self.DEFAULT_WEIGHT_LOCATION}/harmony"
        if not melody_model_weights_location:
            melody_model_weights_location = f"{self.DEFAULT_WEIGHT_LOCATION}/melody"
        self.chord_model_weights_location = chord_model_weights_location
        self.melody_model_weights_location = melody_model_weights_location
        self.filename = filename
        self.harmony = harmony
        self.melody = None

    def write_whole_song(self):
        """Write a whole song from the trained models.

        This means first writing the harmony, then the melody on top of it and finally
        writing out the result as a .xml file.
        """
        self.harmony = self._write_harmony()
        self.melody = self._write_melody()
        self._write_out_as_xml()

    def write_only_melody(self, chords_filename: str):
        """Write only a melody on top of a given chord structure.

        :param chords_filename: The name of the .xml file of the song whose
            chords shall be used.
        """
        parser = HarmonySongParser(chords_filename)
        parser.parse()
        chords = parser.harmony_representation
        harmony_representation = np.array(
            [
                [chord.neural_net_representation for chord in chords],
                [chord.duration for chord in chords],
                [chord.offset for chord in chords],
                np.cumsum([chord.duration for chord in chords]),
            ]
        )
        self.harmony = harmony_representation
        self._write_melody()
        self._write_out_as_xml()

    def _write_harmony(self):
        """Write a harmony (i.e. chords) for the song."""
        model = ChordModel()
        model.compile()
        model.load_weights(f"{self.chord_model_weights_location}/weights")
        input_creator = HarmonyInputCreator(self.key, file=self.filename)
        nn_input = input_creator.get_neural_net_input().astype(float)
        harmony = GreedySearch(
            model=model, start_input=nn_input, nr_measures=self.nr_measures
        ).predict()
        harmony = np.vstack(
            [
                harmony,
                np.cumsum([harmony[1, :]]) - harmony[1, 0],
            ]
        )
        return harmony

    def _write_melody(self):
        model = MelodyModel()
        model.compile()
        model.load_weights(f"{self.chord_model_weights_location}/weights")
        input_creator = MelodyInputCreator(self.key, file=self.filename)
        nn_input = input_creator.get_neural_net_input().astype(float)
        melody = MelodyGreedySearch(
            model=model,
            start_input=nn_input,
            nr_measures=self.nr_measures,
            chords=self.harmony,
        ).predict()
        return melody

    def _write_out_as_xml(self):
        """Write out the resulting song as a .xml file."""
        pass


if __name__ == "__main__":
    SongPredictor("C min7").write_whole_song()
