"""The neural network that will write the harmony."""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GRU

from rnn.model.layers import DenseWithProcessing
from rnn.model.layers import OutputDense


class ChordModel(tf.keras.Model):
    """
    The chord neural network.

    The job of the network is to learn the harmony of jazz songs.
    This is done by giving the network harmony snippets of a given
    length (input_length) and then letting it predict the next
    chord, given the previous chords.

    The architecture of the model is as follows:

    -   There are three inputs: The chords, their durations and their
        offsets.

    -   The chords are passed through an embedding layer.

    -   The embedded chords are concatenated with the (not embedded)
        durations and offsets.

    -   This concatenated information is fed to a Gated Recurrent Unit (GRU).

    -   From here, the network branches into two branches that are both
        essentially multilayer perceptrons: Both branches have one dense
        layer (with relu activation) and one output dense layer (with
        softmax activation).

    :param embedding_dimension: The dimension that the chord embedding
        shall have.
    :param input_length: The length of ONE input sequence that the
        network receives.
    :param embedding_dimension: The number of dimensions to embed
        the chords in.
    :param gru_size: The size of the Gated Recurrent Unit.
    :param dense_size: The size of the dense layer.
    :param gru_dropout_rate: The dropout rate in the Gated Recurrent Unit.
    """

    def __init__(
        self,
        input_length: int = 8,
        dropout_rate: float = 0.4,
        embedding_dimension: int = 24,
        gru_size: int = 256,
        dense_size: int = 128,
        gru_dropout_rate: float = 0.2,
    ) -> None:
        """Initialize the harmony model."""
        super(ChordModel, self).__init__()

        # Embedding
        self.chord_embeddings = Embedding(
            106, embedding_dimension, input_length=input_length
        )

        # Concatenation of embeddings with durations and offsets
        self.concatenate = Concatenate(axis=2)

        # Recurrent unit
        self.batch_norm = BatchNormalization()
        self.gru = GRU(
            gru_size, dropout=gru_dropout_rate, recurrent_dropout=gru_dropout_rate
        )

        # Dense layers before the output layers
        self.dense_chord = DenseWithProcessing(dense_size, dropout_rate)
        self.dense_duration = DenseWithProcessing(dense_size, dropout_rate)

        # Output layers
        self.chord_output = OutputDense(60)
        self.duration_output = OutputDense(49)

    def call(self, inputs: list[np.ndarray]):
        """
        Define the models forward pass.

        :param inputs: The inputs (chords, durations, and offsets).
        :return: The predicted chords and durations.
        """
        chord_input, duration_input, offset_input = inputs
        duration_input = tf.expand_dims(duration_input, axis=2)
        offset_input = tf.expand_dims(offset_input, axis=2)

        embedding = self.chord_embeddings(chord_input)
        x = self.concatenate([embedding, duration_input, offset_input])

        x = self.batch_norm(x)
        gru = self.gru(x)

        dense_chord = self.dense_chord(gru)
        dense_duration = self.dense_duration(gru)

        chord_output = self.chord_output(dense_chord)
        duration_output = self.duration_output(dense_duration)

        return chord_output, duration_output

    def access_model(self):
        """Workaround to plot and summarize the model.

        For some reason, this does not work for the "plain" subclassed
        Keras model; therefore, this method is implemented to access things
        like the number of layers, a plot of the model architecture etc.
        """
        inputs = [tf.keras.Input(shape=8) for _ in range(3)]
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
