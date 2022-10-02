"""A Training pipeline designed to train both models easily."""
import glob
import logging
import os
from multiprocessing import Pool
from typing import Dict
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.callbacks as callbacks
import yaml

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

    :param config: A dictionary read from a .yml file containing all the
        necessary configurations for training the model. The config file
        has to give the following parameters:
        - model_type: A string indicating which model is supposed to be
            trained (either 'melody' or 'harmony').
        - architecture: A dictionary containing all the parameters for
            the model architecture, such as dropout_rate, gru_size etc.
        - training: A dictionary containing all the parameters for the
            training process, such as epochs, validation_split etc.
        - callbacks: A dictionary where each entry is the name of the
            callback (has to be exactly the name of the module in Keras)
            and the value is yet another dictionary containing the
            parameters for the respective callback.
    """

    def __init__(self, config: Dict) -> None:
        """Initialize the training pipeline."""
        self.config = config
        self._model_type = self.config["model_type"]
        if self._model_type == "melody":
            model = MelodyModel
            self.parser = MelodySongParser
        elif self._model_type == "harmony":
            model = ChordModel
            self.parser = HarmonySongParser
        else:
            raise ValueError(
                "Invalid model type given. The only valid options are "
                "'melody' for the melody model and 'harmony' for the chord model."
            )

        # Initialize model with the given parameters.
        self.model = model(**self.config["architecture"])
        logger.debug("Model successfully initialized.")
        self.input_length = self.config["architecture"].get("input_length", 8)
        self.history = None
        self.output_directory = None

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
        logger.debug("Compiling the model.")
        compile_config = self.config["compile_info"]
        self.model.compile(
            optimizer=getattr(
                tf.keras.optimizers, compile_config.get("optimizer", "Adam")
            )(),
            loss={
                # Loss for the main output, i.e. either the harmony or the melody.
                "output_1": getattr(
                    tf.keras.losses, compile_config["loss"][self._model_type]
                )(),
                # Loss for the duration (both models have this second output).
                "output_2": getattr(
                    tf.keras.losses, compile_config["loss"]["duration"]
                )(),
            },
            # We want metrics to be comparable; therefore, they will not be changed
            #   throughout the modeling experimentation.
            metrics=[
                [
                    tf.keras.metrics.SparseCategoricalAccuracy(),
                    tf.keras.metrics.TopKCategoricalAccuracy(k=5),
                ]
                for _ in range(2)
            ],
        )
        logger.info("Model successfully compiled.")
        if compile_config.get("save_architecture_summary", False):
            self.__save_model_summary()
        if compile_config.get("save_architecture_image", False):
            self.__save_model_architecture_as_image()

    def _get_callbacks(self):
        """Get the callbacks for the model training.

        The configs have to be set in the config .yml file; the callbacks are
        then instantiated here with the respective configs.
        """
        return [
            getattr(callbacks, name)(**configs)
            for name, configs in self.config["callbacks"].items()
        ]

    def train(
        self,
        epochs: int = 40,
        validation_split: float = 0.2,
        save_weights: bool = True,
        weights_file_location: str = None,
        previous_weights_path: str = None,
    ) -> None:
        """
        Train the neural net.

        :param epochs: The number of epochs to train the neural net for.
        :param validation_split: THe validation split to apply.
        :param save_weights: Whether or not to save the weights at the end
            of the training process. Defaults to True.
        :param weights_file_location: The location of the file to place the
            weights in.
        :param previous_weights_path: If specified, pre-trained model weights
            will be loaded from the specified path.
        """
        x, y = self._load_training_data()
        self._load_and_compile_model()

        if save_weights:
            if weights_file_location is None:
                path = f"./model/trained_models/{self._model_type}"
                os.makedirs(path, exist_ok=True)
                weights_file_location = f"{path}/weights"
            logger.info(f"The trained model will be saved to {weights_file_location}.")

        if previous_weights_path is not None:
            self.model.load_weights(previous_weights_path)

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

    def __create_output_folder(self):
        directory = "./model/architecture"
        os.makedirs(directory, exist_ok=True)
        self.output_directory = directory

    def __save_model_summary(self):
        """Write out the summary of the model as a .txt file."""
        self.__create_output_folder()
        with open(f"{self.output_directory}/chord_model.txt", "w") as fh:
            self.model.access_model().summary(print_fn=lambda x: fh.write(x + "\n"))

    def __save_model_architecture_as_image(self):
        """Save the model architecture to a .png image."""
        self.__create_output_folder()
        tf.keras.utils.plot_model(
            self.model.access_model(),
            f"{self.output_directory}/harmony_model.png",
            show_shapes=True,
            show_dtype=True,
        )
