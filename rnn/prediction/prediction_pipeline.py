"""A pipeline for predicting, i.e. for writing a song."""
import numpy as np

from rnn.model.chord_model import ChordModel
from rnn.model.melody_model import MelodyModel
from rnn.prediction.input_creator import HarmonyInputCreator
from rnn.prediction.input_creator import MelodyInputCreator


class PredictionPipeline:
    """Pipeline for predicting from a trained model."""

    def __init__(
        self,
        key: str,
        nr_measures: int = 32,
    ) -> None:
        """Initialize the prediction pipeline."""
        self.key = key
        self.nr_measures = nr_measures
        self.input_creator = None
        self.model = None

    def predict(self):
        """Predict from the trained model."""
        nn_input = self.input_creator.get_neural_net_input()
        start_input = list(np.expand_dims(nn_input, 1).astype(float))
        return start_input
        # y = BeamSearch(
        #     self.model, self.key, self.nr_measures, start_input
        # )
        # return y

    def write_out_as_xml(self):
        """Write out the resulting song as a .xml file."""


class ChordModelPredictionPipeline(PredictionPipeline):
    """Prediction pipeline for the chord model."""

    def __init__(self, key: str, weights_location: str, nr_measures: int = 32) -> None:
        """Initialize the chord prediction pipeline."""
        super().__init__(key, nr_measures)
        self.model = ChordModel()
        self.model.load_weights(weights_location)
        self.input_creator = HarmonyInputCreator(self.key)
        # self.chord_model.load_weights("../model/trained_models/harmony/weights")


class MelodyModelPredictionPipeline(PredictionPipeline):
    """Preiction pipeline for the melody model."""

    def __init__(self, key: str, weights_location: str, nr_measures: int = 32) -> None:
        """Initialize the chord prediction pipeline."""
        super().__init__(key, nr_measures)
        self.model = MelodyModel()
        self.model.load_weights(weights_location)
        self.input_creator = MelodyInputCreator(self.key)
        # self.chord_model.load_weights("../model/trained_models/harmony/weights")