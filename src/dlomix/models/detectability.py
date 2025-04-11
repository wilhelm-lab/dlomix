import tensorflow as tf

from ..constants import CLASSES_LABELS, padding_char


class DetectabilityModel(tf.keras.Model):
    def __init__(
        self,
        num_units,
        num_classes=len(CLASSES_LABELS),
        name="autoencoder",
        padding_char=padding_char,
        **kwargs
    ):
        super(DetectabilityModel, self).__init__(name=name, **kwargs)

        self.num_units = num_units
        self.num_classes = num_classes
        self.padding_char = padding_char
        self.alphabet_size = len(padding_char)
        self.one_hot_encoder = tf.keras.layers.Lambda(
            lambda x: tf.one_hot(tf.cast(x, "int32"), depth=self.alphabet_size)
        )
        self.encoder = Encoder(self.num_units)
        self.decoder = Decoder(self.num_units, self.num_classes)

    def call(self, inputs):
        onehot_inputs = self.one_hot_encoder(inputs)
        enc_outputs, enc_state_f, enc_state_b = self.encoder(onehot_inputs)

        dec_outputs = tf.concat([enc_state_f, enc_state_b], axis=-1)

        decoder_inputs = {
            "decoder_outputs": dec_outputs,
            "state_f": enc_state_f,
            "state_b": enc_state_b,
            "encoder_outputs": enc_outputs,
        }

        decoder_output = self.decoder(decoder_inputs)

        return decoder_output


class Encoder(tf.keras.layers.Layer):
    def __init__(self, units, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.units = units

        self.mask_enco = tf.keras.layers.Masking(mask_value=padding_char)

        self.encoder_gru = tf.keras.layers.GRU(
            self.units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

        self.encoder_bi = tf.keras.layers.Bidirectional(self.encoder_gru)

    def call(self, inputs):
        mask_ = self.mask_enco.compute_mask(inputs)

        mask_bi = self.encoder_bi.compute_mask(inputs, mask_)

        encoder_outputs, encoder_state_f, encoder_state_b = self.encoder_bi(
            inputs, initial_state=None, mask=mask_bi
        )

        return encoder_outputs, encoder_state_f, encoder_state_b


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, name="attention_layer", **kwargs):
        super(BahdanauAttention, self).__init__(name=name, **kwargs)
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, inputs):
        query = inputs["query"]
        values = inputs["values"]

        query_with_time_axis = tf.expand_dims(query, axis=1)

        query_values = tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values))
        scores = self.V(query_values)

        attention_weights = tf.nn.softmax(scores, axis=1)

        context_vector = attention_weights * values

        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector


class Decoder(tf.keras.layers.Layer):
    def __init__(self, units, num_classes, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.units = units
        self.num_classes = num_classes

        self.decoder_gru = tf.keras.layers.GRU(
            self.units, return_state=True, recurrent_initializer="glorot_uniform"
        )

        self.attention = BahdanauAttention(self.units)

        self.decoder_bi = tf.keras.layers.Bidirectional(self.decoder_gru)

        self.decoder_dense = tf.keras.layers.Dense(
            self.num_classes, activation=tf.nn.softmax
        )

    def call(self, inputs):
        decoder_outputs = inputs["decoder_outputs"]
        state_f = inputs["state_f"]
        state_b = inputs["state_b"]
        encoder_outputs = inputs["encoder_outputs"]

        states = [state_f, state_b]

        attention_inputs = {"query": decoder_outputs, "values": encoder_outputs}

        context_vector = self.attention(attention_inputs)

        context_vector = tf.expand_dims(context_vector, axis=1)

        x = context_vector

        (
            decoder_outputs,
            decoder_state_forward,
            decoder_state_backward,
        ) = self.decoder_bi(x, initial_state=states)

        x = self.decoder_dense(decoder_outputs)
        # x = tf.expand_dims(x, axis = 1)
        return x
