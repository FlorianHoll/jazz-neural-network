"""Custom layers for the chord and melody networks.

Since both networks function similarly, these layers can be reused
for both networks.
"""
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import L1L2


class DenseWithProcessing(Layer):
    """
    A dense layer with batch norm before the activation and dropout after it.

    This layer will be used after the Gated Recurrent Unit in the two
    branches of the network - one is responsible for predicting the durations
    of the chords/ notes; one is responsible for predicting the actual
    chords/ notes.

    :param dense_size: The size of the dense layer.
        Note: Since we are using relu, the HeNormal() kernel initializer
        will be used (see https://arxiv.org/pdf/1502.01852.pdf).
    :param dropout_rate: The dropout rate to be applied after the dense layer.
        Note: Since the activation is relu, the placement of the dropout
        (before or after the activation) does not matter.
    :param l1_regularization: The L1 regularization rate to be applied
        to the dense layer.
    :param l2_regularization: The L2 regularization rate to be applied
        to the dense layer.
    """

    def __init__(
        self,
        dense_size: int = 128,
        dropout_rate: float = 0.4,
        l1_regularization: float = 1e-4,
        l2_regularization: float = 1e-3,
    ):
        """Initialize the layer."""
        super().__init__()
        self.batch_norm = BatchNormalization()
        self.dense = Dense(
            dense_size,
            activation="relu",
            kernel_initializer=HeNormal(),
            kernel_regularizer=L1L2(l1=l1_regularization, l2=l2_regularization),
        )
        self.dropout = Dropout(dropout_rate)

    def call(self, input_tensor, training=False):
        """Define the forward pass of the layer."""
        x = self.batch_norm(input_tensor, training=training)
        x = self.dense(x)
        x = self.dropout(x)
        return x


class OutputDense(Layer):
    """
    A dense layer with batch norm before the activation and dropout after it.

    This layer will be used for the output of both networks. Since both networks
        have to perform a classification task (finding the right note/ chord
        among many possible ones), all dense output layers will have a softmax
        activation function.

    :param dense_size: The size of the dense layer.
        Note: For the weight initialization, we are using the GlorotNormal()
        initialization method (see
        http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf).
    :param l1_regularization: The L1 regularization rate to be applied
        to the dense layer.
    :param l2_regularization: The L2 regularization rate to be applied
        to the dense layer.
    """

    def __init__(
        self,
        dense_size: int,
        l1_regularization: float = 1e-4,
        l2_regularization: float = 1e-3,
    ):
        """Initialize the layer."""
        super().__init__()
        self.batch_norm = BatchNormalization()
        self.dense = Dense(
            dense_size,
            activation="softmax",
            kernel_initializer=GlorotNormal(),
            kernel_regularizer=L1L2(l1=l1_regularization, l2=l2_regularization),
        )

    def call(self, input_tensor, training=False):
        """Define the forward pass."""
        x = self.batch_norm(input_tensor, training=training)
        x = self.dense(x)
        return x
