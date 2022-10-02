"""The neural network that will write the melody."""
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GRU

from rnn.model.layers import DenseWithProcessing
from rnn.model.layers import OutputDense


class MelodyModel(tf.keras.Model):
    """Melody model."""

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
        super(MelodyModel, self).__init__()

        # Embedding
        self.pitch_embedding = Embedding(
            106, embedding_dimension, input_length=input_length
        )

        # Concatenation of embeddings with durations and offsets
        self.concatenate = Concatenate(axis=2)

        # Recurrent unit
        self.batch_norm = BatchNormalization()
        self.gru = GRU(
            gru_size, dropout=gru_dropout_rate, recurrent_dropout=gru_dropout_rate
        )

        self.dense_chord = DenseWithProcessing(dense_size, dropout_rate)
        self.dense_duration = DenseWithProcessing(dense_size, dropout_rate)

        # Output layers
        self.melody_output = OutputDense(60)
        self.duration_output = OutputDense(49)

    def call(self, inputs):
        """
        Call the model, i.e. its models forward pass.

        :param inputs: The inputs (chords, durations, offsets, and chord notes).
        :return: The predicted melody notes and durations.
        """
        pitch_input, duration_input, offset_input, note1, note2, note3, note4 = inputs

        chord_notes = [note1, note2, note3, note4]

        duration_input = tf.expand_dims(duration_input, axis=2)
        offset_input = tf.expand_dims(offset_input, axis=2)

        embedding = self.pitch_embedding(pitch_input)

        # The chord notes are fed through the same embedding as the pitch heights
        #   since they both are notes and have a very close relationship (mostly,
        #   melody notes are notes that are also in the chord).
        note1_emb, note2_emb, note3_emb, note4_emb = [
            self.pitch_embedding(note) for note in chord_notes
        ]

        x = self.concatenate(
            [
                embedding,
                duration_input,
                offset_input,
                note1_emb,
                note2_emb,
                note3_emb,
                note4_emb,
            ]
        )

        x = self.batch_norm(x)
        gru = self.gru(x)

        dense_chord = self.dense_chord(gru)
        dense_duration = self.dense_duration(gru)

        chord_output = self.melody_output(dense_chord)
        duration_output = self.duration_output(dense_duration)

        return chord_output, duration_output

    def access_model(self):
        """Workaround to plot and summarize the model.

        Subclassed models in Keras cannot be accessed as easily; therefore,
        therefore, this method is implemented to access things like the
        number of layers, a plot of the model architecture etc.
        """
        inputs = [tf.keras.Input(shape=8) for _ in range(7)]
        return tf.keras.Model(inputs=inputs, outputs=self.call(inputs))
