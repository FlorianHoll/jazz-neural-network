"""A Training pipeline designed to train both models easily."""
import glob
import logging
import os
from multiprocessing import Pool
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

from rnn.model.chord_model import ChordModel
from rnn.model.melody_model import MelodyModel
from rnn.music.song import HarmonySongParser
from rnn.music.song import MelodySongParser

logging_formatter = logging.Formatter(
    "%(asctime)s|%(levelname)-8s|%(filename)-25s|%(lineno)-4s|%(message)s"
)
handler = logging.StreamHandler()
handler.setFormatter(logging_formatter)

# Add handler and set level.
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _parse_one_song(
    filename: str, parser: Union[MelodySongParser, HarmonySongParser], input_length: int
) -> np.ndarray:
    """Parse one song (wrapper function).

    The function will then be passed to the multiprocess pool.
    Therefore, it is refactored as a separate function.

    :param filename: The name of the file to be parsed.
    :return: The parsed and converted file as numpy array.
    """
    return parser(filename).parse_and_return_neural_net_input(input_length)


class TrainingPipeline:
    """A training pipeline.

    :param model_to_train: The model to train. Either "melody" or "harmony".
    :param input_length: The length of one input sequence to the network.
    :param dropout_rate: The dropout rate to be applied to the dense layers.
    :param embedding_dimension: The embedding dimension of the notes/ chords.
    :param gru_size: The size of the Gated Recurrent Unit.
    :param dense_size: The size of the dense layers.
    :param gru_dropout_rate: The dropout rate for the GRU.
    """

    def __init__(
        self,
        model_to_train: str,
        input_length: int = 8,
        dropout_rate: float = 0.4,
        embedding_dimension: int = 24,
        gru_size: int = 256,
        dense_size: int = 128,
        gru_dropout_rate: float = 0.2,
    ) -> None:
        """Initialize the training pipeline."""
        self.model_type = model_to_train
        if self.model_type == "melody":
            model = MelodyModel
            self.parser = MelodySongParser
        elif self.model_type == "harmony":
            model = ChordModel
            self.parser = HarmonySongParser
        else:
            raise ValueError(
                "Invalid model type given. The only valid options are "
                "'melody' for the melody model and 'harmony' for the chord model."
            )

        # Initialize model with the given parameters.
        self.model = model(
            input_length=input_length,
            dropout_rate=dropout_rate,
            embedding_dimension=embedding_dimension,
            gru_size=gru_size,
            dense_size=dense_size,
            gru_dropout_rate=gru_dropout_rate,
        )
        logger.debug("Model initialized successfully.")
        self.input_length = input_length
        self.history = None

    def _load_training_data(
        self, parallelize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load the training data.

        :param parallelize: Parallelize the parsing of the training songs?
        :return: The neural net input sequences (x) and the target (y).
        """
        files = glob.glob("../data/*.xml")

        logger.info("Starting to parse all the training songs.")
        if parallelize:
            tasks = [(file, self.parser, self.input_length) for file in files]
            with Pool(os.cpu_count()) as pool:
                nn_input = pool.starmap(_parse_one_song, tasks, chunksize=5)
        else:
            nn_input = [
                self.parser(file).parse_and_return_neural_net_input(self.input_length)
                for file in files
            ]

        nn_input = np.concatenate(nn_input, axis=1)

        x = nn_input[:, :, :-1].astype(np.float)
        y = nn_input[:2, :, -1].astype(np.float)
        return x, y

    def _load_and_compile_model(self):
        """Load and compile the model."""
        nr_output_layers = len(self.model.access_model().outputs)
        logger.debug("Compiling the model.")
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss={
                f"output_{i}": tf.keras.losses.SparseCategoricalCrossentropy()
                for i in range(1, nr_output_layers + 1)
            },
            metrics=[
                [
                    tf.keras.metrics.SparseCategoricalAccuracy(),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=5),
                ]
                for _ in range(nr_output_layers)
            ],
        )
        logger.info("Model successfully compiled.")

    def _get_callbacks(self):
        """Get the callbacks for the model training."""
        callbacks = [
            ModelCheckpoint(
                filepath=(
                    f"./model/partly_trained_models/{self.model_type}/"
                    "weights/weights-{epoch:04d}.ckpt"
                ),
                monitor="val_output_1_sparse_categorical_accuracy",
                verbose=0,
            ),
            ReduceLROnPlateau(
                monitor="output_1_sparse_categorical_accuracy",
                patience=7,
                factor=0.3,
                min_lr=5e-5,
            ),
            EarlyStopping(
                monitor="output_1_sparse_categorical_accuracy",
                patience=10,
                restore_best_weights=True,
            ),
        ]
        return callbacks

    def train(
        self,
        epochs: int = 40,
        validation_split: float = 0.2,
        save_weights: bool = True,
        weights_file_location: str = None,
    ) -> None:
        """
        Train the neural net.

        :param epochs: The number of epochs to train the neural net for.
        :param validation_split: THe validation split to apply.
        :param save_weights: Whether or not to save the weights at the end
            of the training process. Defaults to True.
        :param weights_file_location: The location of the file to place the
            weights in.
        """
        if weights_file_location is None:
            weights_file_location = f"./model/trained_models/{self.model_type}/weights"
        logger.debug(f"Saving model to {weights_file_location}.")

        x, y = self._load_training_data()
        self._load_and_compile_model()
        logger.debug("Starting to train the model.")
        self.history = self.model.fit(
            list(x),
            list(y),
            epochs=epochs,
            validation_split=validation_split,
            callbacks=self._get_callbacks(),
        )
        if save_weights:
            self.model.save_weights(weights_file_location)
        self.plot_training_history()

    def plot_training_history(self) -> None:
        """Plot the training history."""
        if self.history is None:
            return
        acc = self.history.history["loss"]
        val_acc = self.history.history["val_loss"]
        epochs = len(acc)
        plt.plot(range(epochs), acc, "y", label="Training acc")
        plt.plot(range(epochs), val_acc, "r", label="Validation acc")
        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
