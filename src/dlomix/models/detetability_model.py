# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, gaussian_kde
from sklearn.preprocessing import MinMaxScaler
from detectability_model_constants import CLASSES_LABELS, padding_char

class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, units, name = "encoder", **kwargs):
        
        super(Encoder, self).__init__(name = name, **kwargs)
        
        self.units = units
        
        self.mask_enco = tf.keras.layers.Masking(mask_value = padding_char)
        
        self.encoder_gru = tf.keras.layers.GRU(self.units,
                                               return_sequences = True,
                                               return_state = True,
                                               recurrent_initializer='glorot_uniform')
        
        self.encoder_bi = tf.keras.layers.Bidirectional(self.encoder_gru)
        
    def call(self, inputs): 
        
        mask_ = self.mask_enco.compute_mask(inputs)
        
        mask_bi = self.encoder_bi.compute_mask(inputs, mask_)
        
        encoder_outputs, encoder_state_f, encoder_state_b = self.encoder_bi(inputs,
                                                                            initial_state = None,
                                                                            mask = mask_bi) 
        
    
        return encoder_outputs, encoder_state_f,  encoder_state_b
    
    
class BahdanauAttention(tf.keras.layers.Layer):
    
    def __init__(self, units, name = 'attention_layer', **kwargs):
        
        super(BahdanauAttention, self).__init__(name=name, **kwargs)
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
    
    def call(self, inputs):
        
        query = inputs['query']
        values = inputs['values']
        
        query_with_time_axis = tf.expand_dims(query, axis = 1) 
        
        scores = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values))) 
        
        attention_weights = tf.nn.softmax(scores, axis = 1) 
      
        context_vector = attention_weights * values 
        
        context_vector = tf.reduce_sum(context_vector, axis = 1) 
        
        return context_vector 
    
class Decoder(tf.keras.layers.Layer):
    
    def __init__(self, units, num_classes, name="decoder", **kwargs):
        
        super(Decoder, self).__init__(name=name, **kwargs)
        self.units = units
        self.num_classes = num_classes      
        
        
        self.decoder_gru = tf.keras.layers.GRU(self.units,
                                               return_state = True,
                                               recurrent_initializer='glorot_uniform')
         
        self.attention = BahdanauAttention(self.units)
        
        self.decoder_bi = tf.keras.layers.Bidirectional(self.decoder_gru)
        
        self.decoder_dense = tf.keras.layers.Dense(self.num_classes, activation = tf.nn.softmax) 
       
    def call(self, inputs):
        
        decoder_outputs = inputs['decoder_outputs'] 
        state_f = inputs['state_f']
        state_b = inputs['state_b']
        encoder_outputs= inputs['encoder_outputs']
        
        states = [state_f, state_b]
        
        attention_inputs = {'query': decoder_outputs, 'values': encoder_outputs}
        
        context_vector = self.attention(attention_inputs)
        
        context_vector = tf.expand_dims(context_vector, axis = 1)

        x = context_vector
        
        decoder_outputs, decoder_state_forward, decoder_state_backward = self.decoder_bi(x, initial_state = states)
        
        x = self.decoder_dense(decoder_outputs)
        x = tf.expand_dims(x, axis = 1)
        
        return x 
    
class detetability_model(tf.keras.Model):
    
    def __init__(
        self,
        num_units,  
        num_clases = len(CLASSES_LABELS),
        name="autoencoder",
        **kwargs
    ):
    
        super(detetability_model, self).__init__(name=name, **kwargs)

        self.num_units = num_units
        self.num_clases = num_clases
        self.encoder = Encoder(self.num_units) 
        self.decoder = Decoder(self.num_units, self.num_clases)

    def call(self, inputs):
        
        enc_outputs, enc_state_f, enc_state_b = self.encoder(inputs)
        
        
        dec_outputs = tf.concat([enc_state_f, enc_state_b], axis = -1)
 
        decoder_inputs = {'decoder_outputs': dec_outputs,
                          'state_f': enc_state_f,
                          'state_b': enc_state_b,
                          'encoder_outputs':enc_outputs}
        
        decoder_output = self.decoder(decoder_inputs)
        
        return decoder_output