# -*- coding: utf-8 -*-
"""
TODO: 
Create new head to predict mz and ab
"""

from copy import deepcopy
import numpy as np
import tensorflow as tf
K = tf.keras
L = K.layers
A = K.activations
I = K.initializers
gelu = tf.keras.activations.gelu
twopi = 2*np.pi

def get_norm_type(string):
    if string=='layer':
        return L.LayerNormalization
    elif string=='batch':
        return L.BatchNormalization

class QKVAttention(L.Layer):
    def __init__(self, heads, is_relpos=False, max_rel_dist=None):
        super(QKVAttention, self).__init__()
        self.heads = heads
        self.is_relpos = is_relpos
        self.maxd = max_rel_dist
    
    def build_relpos_tensor(self, seq_len, maxd=None):
        # This is the equivalent of an input dependent ij bias, rather than one
        # that is a direclty learned ij tensor
        maxd = seq_len-1 if maxd==None else (maxd-1 if maxd==seq_len else maxd)
        a = tf.range(seq_len, dtype=tf.int32)
        b = tf.range(seq_len, dtype=tf.int32)
        relpos = a[:,None] - b[None]
        tsr = tf.random.normal((2*seq_len-1, self.dim), 0, seq_len**-0.5)
        # set maximum distance
        relpos = tf.clip_by_value(relpos, -maxd, maxd)
        relpos += maxd
        tsr = tf.nn.embedding_lookup(tsr, relpos)
        relpos_tsr = tf.Variable(tsr, trainable=True)
        
        return relpos_tsr
    
    def build(self, x):
        self.sl = x[1]
        self.dim = x[-1]
        self.scale = 1 / self.dim**0.25
        if self.is_relpos:
            self.ak = self.build_relpos_tensor(x[1], self.maxd) # sl, sl, dim
            self.av = self.build_relpos_tensor(x[1], self.maxd) # sl, sl, dim
    
    def call(self, Q, K, V, mask=None):
        # shape: batch/heads/etc, sequence_length, dim
        QK = tf.einsum('abc,adc->abd', self.scale*Q, self.scale*K)
        if self.is_relpos:
            QK += tf.einsum('abc,bec->abe', Q, self.ak)
        QK = tf.reshape(QK, (-1, self.heads, self.sl, K.shape[1]))
        
        # mask.shape: bs, 1, 1, sl
        mask = tf.zeros_like(QK) if mask==None else mask[:,None,None,:]
        weights = A.softmax(QK-mask, axis=-1)
        weights = tf.reshape(weights, (-1, self.sl, V.shape[1]))
        
        att = tf.einsum('abc,acd->abd', weights, V)
        if self.is_relpos:
            att += tf.einsum('abc,bcd->abd', weights, self.av)
        
        return att

class SelfAttention(L.Layer):
    def __init__(self, d, h, out_units=None, alphabet=False, dropout=0):
        super(SelfAttention, self).__init__()
        self.d = d
        self.h = h
        self.out_units = out_units
        self.alphabet = alphabet
        
        self.attention_layer = QKVAttention(h)
        self.drop = L.Dropout(dropout)
        if alphabet:
            self.alpha = tf.Variable(1.0, trainable=True)
            self.beta = tf.Variable(0.0, trainable=True)
    
    def get_qkv(self, qkv):
        Q, K, V = tf.split(qkv, 3, axis=-1) # per: bs, sl, d*h 
        Q = tf.reshape(Q, (-1, self.sl, self.d, self.h)) # bs, sl, d, h
        Q = tf.reshape(tf.transpose(Q, (0,3,1,2)), (-1, self.sl, self.d)) 
        K = tf.reshape(K, (-1, self.sl, self.d, self.h)) # bs, sl, d, h
        K = tf.reshape(tf.transpose(K, (0,3,1,2)), (-1, self.sl, self.d))
        V = tf.reshape(V, (-1, self.sl, self.d, self.h)) # bs, sl, d, h
        V = tf.reshape(tf.transpose(V, (0,3,1,2)), (-1, self.sl, self.d))
        
        return Q, K, V # bs*h, sl, d
    
    def build(self, x):
        in_units = x[-1]
        self.sl = x[1]
        self.out_units = x[-1] if self.out_units==None else self.out_units
        
        #limit = np.sqrt(6 / (self.d + x[-1]))
        limit = np.sqrt(6 / (in_units*self.d + in_units*self.h))
        self.qkv = L.Dense(
            3*self.d*self.h, use_bias=False, 
            kernel_initializer=I.RandomUniform(-limit, limit)
        )
        
        limit = np.sqrt(6 / (self.h*self.d + self.h*self.out_units))
        init = I.RandomUniform(-limit, limit)
        self.Wo = L.Dense(
            self.out_units, use_bias=False,
            #kernel_initializer=I.RandomNormal(0, 0.3*(self.h*self.d)**-0.5)
            kernel_initializer=init
        )
        self.shortcut = (
            A.linear 
            if self.out_units==x[-1] else 
            L.Dense(self.out_units, use_bias=False)
        )
    
    def call(self, x, mask=None):
        # x.shape; bs, sl, units
        qkv = self.qkv(x) # bs, sl, 3*d*h
        Q, K, V = self.get_qkv(qkv) # bs*h, sl, d
        att = self.attention_layer(Q, K, V, mask) # bs*h, sl,  d
        att = tf.reshape( 
            tf.transpose(
                tf.reshape(att, (-1, self.h, self.sl, self.d)), (0,2,3,1)
            ), (-1, self.sl, self.d*self.h)
        ) # bs, sl, d*h
        resid = self.Wo(att) # bs, sl, out_units
        resid = self.drop(resid)
        if self.alphabet:
            output = self.alpha*self.shortcut(x) + self.beta*resid
        else:
            output = self.shortcut(x) + resid
        
        return output

class CrossAttention(L.Layer):
    def __init__(self, d, h, out_units=None, dropout=0):
        super(CrossAttention, self).__init__()
        self.d = d
        self.h = h
        self.out_units = out_units
        
        self.Wq = L.Dense(d*h, use_bias=False)
        self.Wkv = L.Dense(2*d*h, use_bias=False)
        
        self.attention_layer = QKVAttention(h)

        self.drop = L.Dropout(dropout)
    
    def get_qkv(self, q, kv):
        bs, sl, units = q.shape
        bs, sl2, units = kv.shape
        Q = tf.reshape(q, (bs, sl, self.d, self.h))
        Q = tf.reshape(tf.transpose(Q, [0,3,1,2]), (-1, sl, self.d))
        K, V = tf.split(kv, 2, -1)
        K = tf.reshape(K, (bs, sl2, self.d, self.h))
        K = tf.reshape(tf.transpose(K, [0,3,1,2]), (-1, sl2, self.d))
        V = tf.reshape(V, (bs, sl2, self.d, self.h))
        V = tf.reshape(tf.transpose(V, [0,3,1,2]), (-1, sl2, self.d))
        
        return Q, K, V
    
    def build(self, x):
        self.sl = x[1]
        self.out_units = x[-1] if self.out_units==None else self.out_units
        self.Wo = L.Dense(
            self.out_units, use_bias=False,
            kernel_initializer=I.RandomNormal(0, 0.3*(self.h*self.d)**-0.5)
        )
        self.shortcut = (
            A.linear
            if self.out_units==x[-1] else
            L.Dense(self.out_units, use_bias=False)
        )
    
    def call(self, q_feats, kv_feats, mask=None):
        Q = self.Wq(q_feats)
        KV = self.Wkv(kv_feats)
        Q, K, V = self.get_qkv(Q, KV)
        att = self.attention_layer(Q, K, V, mask)
        att = tf.reshape(
            tf.transpose(
                tf.reshape(att, (-1, self.h, self.sl, self.d)), [0,2,1,3]
            ), (-1, self.sl, self.h*self.d)
        )
        resid = self.Wo(att)
        resid = self.drop(resid)
        
        return self.shortcut(q_feats) + resid

class FFT(L.Layer):
    def __init__(self):
        super(FFT, self).__init__()

    def call(self, x):
         bs, sl, ch = x.shape
         return tf.cast(
             tf.transpose(
                 tf.signal.fft(
                     tf.transpose(
                         tf.signal.fft(tf.cast(x, tf.complex64)), 
                         [0,2,1]
                     )
                 ), [0,2,1]
             ), tf.float32
         ) / (sl*ch)

class FFTLayer(L.Layer):
    def __init__(self, alphabet=False):
        super(FFTLayer, self).__init__()
        self.mixing_layer = FFT()
        self.alphabet = alphabet
        if alphabet:
            self.alpha = tf.Variable(1., trainable=True)
            self.beta = tf.Variable(0., trainable=True)

    def call(self, x, **kwargs):
        mixing_output = self.mixing_layer(x)
        if self.alphabet:
            output = self.alpha*x + self.beta*mixing_output
        else:
            output = x + mixing_output

        return output

class FFN(L.Layer):
    def __init__(self, unit_multiplier=1, out_units=None, alphabet=False, dropout=0):
        super(FFN, self).__init__()
        self.mult = unit_multiplier
        self.out_units = out_units
        self.alphabet = alphabet

        self.dropout1 = L.Dropout(dropout)
        self.dropout2 = L.Dropout(dropout)
        if alphabet:
            self.alpha = tf.Variable(1.0, trainable=True)
            self.beta = tf.Variable(0.0, trainble=True)
    
    def build(self, x):
        self.out_units = x[-1] if self.out_units==None else self.out_units
        self.W1 = L.Dense(int(x[-1]*self.mult))
        self.W2 = L.Dense(
            self.out_units, use_bias=False, 
            kernel_initializer=I.RandomNormal(0, 0.1*(x[-1]*self.mult)**-0.5)
        )
    
    def call(self, x, embed=None):
        out = self.W1(x)
        out = self.dropout1(A.relu(out + (0 if embed==None else embed)))
        out = self.dropout2(self.W2(out))
        if self.alphabet:
            output = self.alpha*x + self.beta*out
        else:
            output = x + out
        
        return output

class TransBlock(L.Layer):
    def __init__(self,
                 attention_dict,
                 ffn_dict,
                 norm_type='layer', 
                 prenorm=False,
                 use_embed=False,
                 preembed=True,
                 postembed=True,
                 is_cross=False
    ):
        super(TransBlock, self).__init__()
        self.norm_type = norm_type
        self.mult = ffn_dict['unit_multiplier']
        self.prenorm = prenorm
        self.use_embed = use_embed
        self.preembed = preembed
        self.postembed = postembed
        self.is_cross = is_cross
        
        if preembed: self.alpha = tf.Variable(0.1, trainable=True)
        norm = get_norm_type(norm_type)
        
        self.norm1 = norm()
        self.norm2 = norm()

        
        self.selfattention = SelfAttention(**attention_dict)
        if is_cross:
            self.crossnorm = norm()
            self.crossattention = CrossAttention(**attention_dict)
        self.ffn = FFN(**ffn_dict)
    
    def build(self, x):
        if self.use_embed:
            units = x[-1] if self.preembed else int(x[-1]*self.mult)
            self.embed = L.Dense(units)
    
    def call(self, x, kv_feats=None, temb=None, spec_mask=None, seq_mask=None):
        selfmask = seq_mask if self.is_cross else spec_mask
        Temb = self.embed(temb)[:,None,:] if self.use_embed else 0
        
        out = x + self.alpha*Temb if self.preembed else x           # alpha used as "gate"
        out = self.norm1(out) if self.prenorm else out
        out = self.selfattention(out, mask=selfmask)
        if self.is_cross:
            out = self.crossnorm(out) if self.prenorm else out
            out = self.crossattention(out, kv_feats, spec_mask)
            out = out if self.prenorm else self.crossnorm(out)
        out = self.norm2(out) if self.prenorm else self.norm1(out)
        out = self.ffn(out, Temb) if self.postembed else self.ffn(out, None)
        out = out if self.prenorm else self.norm2(out)
        
        return out

class ResBlock(L.Layer):
    """
    residual block
    """
    def __init__(self, out_channels, ks, pad):  # use config
        """
        initialize residual module

        :param config: configuration
        :param outch: number of output channels
        """
        super(ResBlock, self).__init__()
        self.conv1 = L.Conv1D(out_channels, ks, strides=1, padding='same', use_bias=False)
        self.norm1 = L.BatchNormalization()
        self.act = tf.nn.relu
        self.conv2 = nn.Conv1D(out_channels, ks, strides=1, padding='same', use_bias=False)
        self.norm2 = nn.BatchNormalization()
        self.shortcut = L.Conv1D(out_channels, 1, strides=1, padding='same') if inch != out_channels else A.linear

    def forward(self, inp):
        """
        forward pass for residual module

        :param inp: input tensor
        :return: the block
        """
        out = self.conv1(inp)
        out = self.norm1(out)
        out = self.act(out)
        return self.act(self.shortcut(inp) + self.norm2(self.conv2(out)))

class CompoundConv(L.Layer):
    """
    compound convolution
    """
    def __init__(self, outch, rang):  # use config
        """
        initialize compound convolution module

        :param config: configurations
        :param outch: output channels
        :param rang: list of filter sizes
        """
        super(CompoundConv, self).__init__()
        self.convs = nn.ModuleList(
            [L.Conv1D(outch, m, strides=1, padding='same') for m in rang]
        )

    def forward(self, inp):
        """
        forward pass of compound convolution module

        :param inp: input tensor
        :return: output tensor
        """
        return tf.concat([m(inp) for m in self.convs], axis=-1)

class PrecursorToken(L.Layer):
    def __init__(self, units, ff_units, min_lam, max_lam):
        super(PrecursorToken, self).__init__()
        self.FF = lambda t: FourierFeatures(t, min_lam, max_lam, ff_units)
        self.linear = L.Dense(units)

    def call(self, x):
        return self.linear(self.FF(x))

class KerasActLayer(L.Layer):
    def __init__(self, activation):
        super(KerasActLayer, self).__init__()
        self.act = activation
    def call(self, x):
        return self.act(x)

def FourierFeatures(t, min_lam, max_lam, embedsz):
    x = tf.range(embedsz//2, dtype=tf.float32)
    x /= embedsz//2 - 1
    denom = (min_lam / twopi) * (max_lam / min_lam) ** x
    if len(t.shape) == 1:
        t = tf.expand_dims(t, -1)
    embed = t / denom[None]

    return tf.concat([tf.cos(embed), tf.sin(embed)], axis=-1)
