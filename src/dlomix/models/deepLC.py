import tensorflow as tf

from ..constants import ALPHABET_UNMOD


class DeepLCRetentionTimePredictor(tf.keras.Model):
    def __init__(
        self, seq_length=60, alphabet=ALPHABET_UNMOD, use_global_features=False
    ):
        super(DeepLCRetentionTimePredictor, self).__init__()
        self.seq_length = seq_length
        self._use_global_features = use_global_features

        self.leaky_relu = tf.keras.layers.ReLU(max_value=20, negative_slope=0.1)

        self._build_aminoacid_branch()
        self._build_diaminoacid_branch()
        self._build_onehot_encoding_branch()
        self._build_regressor()
        self.output_layer = tf.keras.layers.Dense(1)

        if self._use_global_features:
            self._build_global_features_branch()

    def _build_aminoacid_branch(self):
        self.aminoacid_branch = tf.keras.Sequential(
            [
                self._build_conv_pool_block(n_filters=256, kernel=8, padding="same"),
                self._build_conv_pool_block(n_filters=128, kernel=8, padding="same"),
                self._build_conv_pool_block(
                    n_filters=64, kernel=8, padding="same", pool=False
                ),
                tf.keras.layers.Flatten(),
            ]
        )

    def _build_diaminoacid_branch(self):
        self.diaminoacid_branch = tf.keras.Sequential(
            [
                self._build_conv_pool_block(n_filters=128, kernel=2, padding="same"),
                self._build_conv_pool_block(n_filters=64, kernel=2, padding="same"),
                tf.keras.layers.Flatten(),
            ]
        )

    def _build_global_features_branch(self):
        self.global_features_branch = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(16, activation=self.leaky_relu),
                tf.keras.layers.Dense(16, activation=self.leaky_relu),
                tf.keras.layers.Dense(16, activation=self.leaky_relu),
            ]
        )

    def _build_onehot_encoding_branch(self):
        self.onehot_encoding_branch = tf.keras.Sequential(
            [
                self._build_conv_pool_block(
                    n_filters=2,
                    kernel=2,
                    padding="same",
                    activation="tanh",
                    pool_strides=10,
                    pool_size=10,
                ),
                tf.keras.layers.Flatten(),
            ]
        )

    def _build_regressor(self):
        self.regressor = tf.keras.Sequential(
            [tf.keras.layers.Dense(128, activation=self.leaky_relu) for _ in range(5)]
        )

    def _build_conv_pool_block(
        self,
        n_conv_layers=2,
        n_filters=256,
        kernel=8,
        padding="same",
        activation="leaky_relu",
        pool=True,
        pool_strides=2,
        pool_size=2,
    ):
        # leaky relu by default
        activation_fn = self.leaky_relu

        if activation in ["tanh", "relu"]:
            activation_fn = activation

        block = tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=n_filters,
                    kernel_size=kernel,
                    padding=padding,
                    activation=activation_fn,
                )
                for _ in range(n_conv_layers)
            ]
        )
        if pool:
            block.add(tf.keras.layers.MaxPooling1D(pool_size=2, strides=2))

        return block

    def call(self, inputs, **kwargs):
        outputs = {}

        onehot_encoded = tf.one_hot(inputs["seq"], depth=self.seq_length)

        if self._use_global_features:
            outputs["global_features_output"] = self.global_features_branch(
                inputs["global_features"]
            )

        outputs["onehot_branch_output"] = self.onehot_encoding_branch(onehot_encoded)
        outputs["aminoacids_branch_output"] = self.aminoacid_branch(inputs["counts"])
        outputs["diaminoacids_branch_output"] = self.diaminoacid_branch(
            inputs["di_counts"]
        )

        concatenated_output = tf.concat(outputs.values(), axis=1)
        concatenated_output = self.regressor(concatenated_output)
        return self.output_layer(concatenated_output)
