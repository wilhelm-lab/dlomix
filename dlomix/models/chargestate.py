import tensorflow as tf

from ..constants import ALPHABET_UNMOD

"""
--------------------------------------------------------------
TASK MODELS
TODO: DominantChargeStatePredictor
    -

TODO: ObservedChargeStatePredictor
    -

TODO: ChargeStateProportionPredictor
    -
--------------------------------------------------------------
"""


class DominantChargeStatePredictor(tf.keras.Model):
    """
    Task 1
    Predict the dominant charge state
    Prosite Charge-State Adaption
    """

    def __init__(
        self,
        embedding_dim=16,
        seq_length=30,
        encoder="conv1d",
        vocab_dict=ALPHABET_UNMOD,
        num_classes=6,
    ):
        super(DominantChargeStatePredictor, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(vocab_dict) + 2

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=embedding_dim,
            input_length=seq_length,
        )

        self._build_encoder(encoder)

        self.flatten = tf.keras.layers.Flatten()
        self.regressor = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
            ]
        )

        self.output_layer = tf.keras.layers.Dense(num_classes, activation="softmax")

    def _build_encoder(self, encoder_type):
        if encoder_type.lower() == "conv1d":
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv1D(
                        filters=256, kernel_size=3, padding="same", activation="relu"
                    ),
                    tf.keras.layers.Conv1D(
                        filters=512, kernel_size=3, padding="valid", activation="relu"
                    ),
                    tf.keras.layers.MaxPooling1D(pool_size=2),
                ]
            )
        else:
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.LSTM(256, return_sequences=True),
                    tf.keras.layers.LSTM(256),
                ]
            )

    def call(self, inputs, **kwargs):
        # print("shape of inputs: ", inputs.shape)
        x = self.embedding(inputs)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.regressor(x)
        x = self.output_layer(x)
        # print("shape of output: ", x.shape)

        return x


class ObservedChargeStatePredictor(tf.keras.Model):
    """
    Task 2
    Predict all observed charge states
    """

    def __init__(
        self,
        embedding_dim=16,
        seq_length=30,
        encoder="conv1d",
        vocab_dict=ALPHABET_UNMOD,
        num_classes=6,
    ):
        super(ObservedChargeStatePredictor, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(vocab_dict) + 2

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=embedding_dim,
            input_length=seq_length,
        )

        self._build_encoder(encoder)

        self.flatten = tf.keras.layers.Flatten()
        self.regressor = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
            ]
        )

        self.output_layer = tf.keras.layers.Dense(num_classes, activation="sigmoid")

    def _build_encoder(self, encoder_type):
        if encoder_type.lower() == "conv1d":
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv1D(
                        filters=256, kernel_size=3, padding="same", activation="relu"
                    ),
                    tf.keras.layers.Conv1D(
                        filters=512, kernel_size=3, padding="valid", activation="relu"
                    ),
                    tf.keras.layers.MaxPooling1D(pool_size=2),
                ]
            )
        else:
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.LSTM(256, return_sequences=True),
                    tf.keras.layers.LSTM(256),
                ]
            )

    def call(self, inputs, **kwargs):
        # print("Input shape:", inputs.shape)
        x = self.embedding(inputs)
        # print("Post-embedding shape:", x.shape)
        x = self.encoder(x)
        # print("Post-encoder shape:", x.shape)
        x = self.flatten(x)
        # print("Post-flatten shape:", x.shape)
        x = self.regressor(x)
        # print("Post-regressor shape:", x.shape)
        x = self.output_layer(x)
        # print("Output shape:", x.shape)
        return x


class ChargeStateProportionPredictor(tf.keras.Model):
    """
    Task 3
    Predict the proportion of all observed charge states
    """

    def __init__(
        self,
        embedding_dim=16,
        seq_length=30,
        encoder="conv1d",
        vocab_dict=ALPHABET_UNMOD,
        num_classes=6,
    ):
        super(ChargeStateProportionPredictor, self).__init__()

        # tie the count of embeddings to the size of the vocabulary (count of amino acids)
        self.embeddings_count = len(vocab_dict) + 2

        self.embedding = tf.keras.layers.Embedding(
            input_dim=self.embeddings_count,
            output_dim=embedding_dim,
            input_length=seq_length,
        )

        self._build_encoder(encoder)

        self.flatten = tf.keras.layers.Flatten()
        self.regressor = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
            ]
        )

        self.output_layer = tf.keras.layers.Dense(num_classes, activation="sigmoid")

    def _build_encoder(self, encoder_type):
        if encoder_type.lower() == "conv1d":
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv1D(
                        filters=256, kernel_size=3, padding="same", activation="relu"
                    ),
                    tf.keras.layers.Conv1D(
                        filters=512, kernel_size=3, padding="valid", activation="relu"
                    ),
                    tf.keras.layers.MaxPooling1D(pool_size=2),
                ]
            )
        else:
            self.encoder = tf.keras.Sequential(
                [
                    tf.keras.layers.LSTM(256, return_sequences=True),
                    tf.keras.layers.LSTM(256),
                ]
            )

    def call(self, inputs, **kwargs):
        x = self.embedding(inputs)
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.regressor(x)
        x = self.output_layer(x)
        return x

    pass
