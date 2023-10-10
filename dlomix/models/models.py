# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:31:21 2023

@author: joell
"""
import tensorflow as tf
K = tf.keras
L = K.layers
A = K.activations
I = K.initializers
import numpy as np

nm = 0.03
initEmb = I.RandomNormal(0, nm)
initPos = lambda s: I.random_normal(0, nm)(s)
initQ = lambda s: I.random_normal(0, s[0]**-0.5*s[1]**-0.25)(s)
initK = lambda s: I.random_normal(0, s[0]**-0.5*s[1]**-0.25)(s)
initL = lambda s, eltargstd=0.5: ( 
    I.random_normal(0, eltargstd*s[0]**-0.5)(s)
)
initW = lambda s: I.random_normal(0, 0.01)(s)
initV = lambda s, wtargstd=0.012, sigWw=1, seq_len=40: I.random_normal(
    0, wtargstd**-1 * s[-1]**-0.5 * s[0]**-0.5 * seq_len**-0.5 * sigWw**-1
)(s)
initO = lambda s: I.random_normal(0, nm)(s)
initFFN = lambda s, stdin=1, targ=1: I.random_normal(
    0, targ*s[0]**-0.5*stdin**-1
)(s)
initProj = lambda s: I.RandomNormal(0, s[0]**-0.5)(s)
initFin = I.RandomNormal(0, 0.1)

class TalkingHeads(L.Layer):
    def __init__(self,
                 dk,
                 dv,
                 hv,
                 hk=None,
                 h=None,
                 drop=0,
                 rel_bias=False,
                 out_dim=None
                 ):
        super(TalkingHeads, self).__init__()
        self.dk = dk
        self.dv = dv
        self.hv = hv
        self.hk = hv if hk==None else hk
        self.h = hv if h==None else h
        self.rel_bias = rel_bias
        self.out_dim = out_dim
        
        self.alphaq = tf.Variable(tf.fill((1,), 2.), trainable=True)
        self.alphak = tf.Variable(tf.fill((1,), 2.), trainable=True)
        self.alphav = tf.Variable(tf.fill((1,), 2.), trainable=True)
        
        self.Wl = tf.Variable(initL((self.hk, self.h)), trainable=True)
        self.Ww = tf.Variable(initW((self.h, self.hv)), trainable=True)
        
        self.drop = A.linear if drop==0 else L.Dropout(drop)
    
    def build(self, x):
        self.out_dim = x[-1] if self.out_dim==None else self.out_dim
        self.Wq = tf.Variable(
            initQ((x[-1], self.dk, self.hk)), trainable=True
        )
        self.Wk = tf.Variable(
            initK((x[-1], self.dk, self.hk)), trainable=True
        )
        self.Wv = tf.Variable(
            initV((x[-1], self.dv, self.hv)), trainable=True
        )
        
        self.Wo = tf.Variable(
            initO((self.dv*self.hv, self.out_dim)), trainable=True
        )
        
        self.shortcut = ( 
            L.Dense(self.out_dim) if self.out_dim != x[-1] else A.linear
        )
        
        self.seq_len = x[1]
        self.relative_pos_bias = self.get_relative_pos_bias()
    
    def get_relative_pos_bias(self):
        # coords_1 = tf.range(self.seq_len)[None]
        # coords_2 = tf.range(self.seq_len)[:, None]
        # rel_coords_indices = coords_1-coords_2
        # rel_coords_indices -= tf.reduce_min(rel_coords_indices)
        shape = (self.seq_len, self.seq_len, self.h)
        return ( 
            tf.Variable(tf.zeros(shape, dtype=tf.float32), trainable=True) 
            if self.rel_bias else 
            tf.zeros(shape)
        )
    
    def call(self, inp, mask=None):
        b,s,c = inp.shape
        mask = tf.zeros((0)) if mask==None else mask
        
        Q = A.sigmoid(self.alphaq)*tf.einsum('abc,cde->abde', inp, self.Wq)
        K = A.sigmoid(self.alphak)*tf.einsum('abc,cde->abde', inp, self.Wk)
        V = A.sigmoid(self.alphav)*tf.einsum('abc,cde->abde', inp, self.Wv)
        
        J = tf.einsum('abcd,aecd->abed', Q, K)
        EL = ( 
            tf.einsum('abcd,de->abce', J, self.Wl) + 
            self.relative_pos_bias
        )
        W = tf.nn.softmax(EL, axis=2)
        U = tf.einsum('abcd,de->abce', W, self.Ww)
        O = tf.einsum('abcd,aced->abed', U, V)
        O = tf.reshape(O, (-1, s, self.dv*self.hv))
        resid = self.drop(tf.einsum('abc,cd->abd', O, self.Wo))
        
        Inp = self.shortcut(inp)
        output = Inp + resid
        
        return output

class FFN(L.Layer):
    def __init__(self,
                 units=None,
                 embed=None,
                 learn_embed=True,
                 drop=0,
                 ):
        super(FFN, self).__init__()
        self.units = units
        self.embed = embed
        self.learn_embed = learn_embed
        
        self.drop = A.linear if drop==0 else L.Dropout(drop)
        
    def build(self, x):
        self.units = x[-1] if self.units==None else self.units
        if (self.embed is not None) and (self.learn_embed == False):
            assert self.units==self.embed, (
                "units must be equal to embed dimensions if embed not learned"
            )
        
        self.W1 = tf.Variable(
            initFFN((x[-1], self.units), 1, 1)
        )
        self.W2 = tf.Variable(
            initFFN((self.units, x[-1]), 1, 0.1)
        )
        
        if self.embed is not None:
            self.chce = (
                L.Dense(self.units) 
                if self.learn_embed | self.embed!=self.units else 
                A.linear
            )
    
    def call(self, inp, embed_inp=None):
        emb = 0 if self.embed==None else self.chce(embed_inp)[..., None]
        resid1 = A.relu(tf.einsum('abc,cd->abd', inp, self.W1) + emb)
        resid2 = tf.einsum('abc,cd->abd', resid1, self.W2)
        output = inp + resid2
        
        return output

class TransBlock(L.Layer):
    def __init__(self, 
                 hargs,
                 fargs
                 ):
        super(TransBlock, self).__init__()
        self.out_units_head = hargs[-1]
        self.head = TalkingHeads(*hargs)
        self.ffn = FFN(*fargs)
        
    def build(self, x):
        units1 = x[-1] if self.out_units_head == None else self.out_units_head
        self.out_units_head = units1
        self.norm1 = L.BatchNormalization(epsilon=1e-5)
        self.norm2 = L.BatchNormalization(epsilon=1e-5)
    
    def call(self, inp, embed=None, mask=None):
        out = self.head(inp, mask)
        out = self.norm1(out)
        out = self.ffn(out, embed)
        out = self.norm2(out)
        
        return out

class FlipyFlopy(K.Model):
    def __init__(self,
                 out_dim=7919,
                 embedsz=256,
                 blocks=9,
                 head=(16,16,64),
                 units=None,
                 drop=0,
                 filtlast=512,
                 mask=False,
                 CEembed=False,
                 CEembed_units=256,
                 learn_ffn_embed=True,
                 pos_type='learned',
                 ):
        super(FlipyFlopy, self).__init__()
        self.embedsz = embedsz
        self.blocks = blocks
        self.head_args = head
        self.units = units
        self.mask = mask
        self.CEembed = CEembed
        self.cesz = CEembed_units
        self.drop = drop
        self.learn_ffn_embed = learn_ffn_embed
        self.pos_type = pos_type
        
        #vocab = list('ARNDCQEGHILKMFPSTWYV')
        #self.first = L.StringLookup(vocabulary=vocab, output_mode='int')
        
        self.embed = tf.Variable(initEmb((38, 256)))
        self.embed_norm = L.BatchNormalization(epsilon=1e-5)
        
        self.Proj = tf.Variable(
            initProj((256,512)), trainable=True, name='proj'
        )
        self.ProjNorm = L.BatchNormalization(epsilon=1e-5)
        self.final = L.Dense(
            out_dim, activation='sigmoid', kernel_initializer=initFin
        )
        
    def build(self, x):
        self.seq_len = x[1]#x['sequence'][1]
        self.units = self.embedsz if self.units==None else self.units
        if self.pos_type=='learned':
            self.pos = tf.Variable(
                initPos((self.seq_len, self.embedsz)), trainable=True
            )
        else:
            pos = (
                np.arange(self.seq_len)[:, None] *
                np.exp(
                    -np.log(100) * 
                    np.arange(self.embedsz//2) / 
                    (self.embedsz//2)
                )[None]
            )
            self.pos = tf.Variable(
                tf.constant(
                    np.concatenate([np.cos(pos), np.sin(pos)], axis=-1)
                ), trainable=False
            )
        
        if self.CEembed:
            self.denseCH = L.Dense(self.cesz)
            self.denseCE = L.Dense(self.cesz)
            self.postcat = ( 
                A.linear if self.learn_ffn_embed else L.Dense(self.units)
            )
            ffnembed = 2*self.cesz if self.learn_ffn_embed else self.units
        else:
            ffnembed = None
        
        head_args = tuple(self.head_args) + (None,None,self.drop,None)
        ffn_args = (
            self.units, ffnembed, self.learn_ffn_embed, self.drop
        )
        
        self.main = [
            TransBlock(head_args, ffn_args) 
            for _ in range(self.blocks)
        ]
    
    def embedCE(self, ce, embedsz, freq=10000.):
        # ce.shape = (bs,)
        embed = (
            ce[:, None] *
            tf.exp(
                -tf.math.log(freq) *
                tf.range(embedsz//2, dtype=tf.float32)/(embedsz//2)
            )[None]
        )
        return tf.concat([tf.cos(embed), tf.sin(embed)], axis=-1)
    
    def call(self, inp):
        # if self.CEembed:
        #     # unpack input
        #     [inp, inpch, inpce] = [
        #         inp['sequence'], 
        #         inp['precursor_charge'], 
        #         inp['collision_energy']
        #     ]
            
        #     inpch = tf.cast(tf.argmax(inpch, axis=-1)+1, tf.float32)
        #     ch_embed = A.swish(
        #         self.denseCH(self.embedCE(inpch, self.cesz, 10))
        #     )
        #     inpce = tf.squeeze(inpce)
        #     ce_embed = A.swish(
        #         self.denseCE(self.embedCE(inpce, self.cesz, 100))
        #     )
        #     embed = self.postcat(tf.concat([ch_embed, ce_embed], axis=-1))
        # else:
        #     seq = tf.one_hot(self.first(inp['sequence']), depth=21)
        #     print("INPUT KEYS: ", inp.keys())
        #     charge = tf.one_hot(tf.cast(inp['precursor_charge'], tf.int32), depth=8)
        #     charge = tf.tile(
        #         charge[:,None,:], [1, self.seq_len, 1]
        #     )
        #     energy = tf.tile(
        #         inp['collision_energy'][:,None,:], [1, self.seq_len, 1]
        #     )
        #     inp = tf.concat([seq, charge, energy], axis=-1)
        #     embed = None
        embed=None
        out = tf.einsum("abc,cd->abd", inp, self.embed)
        out += self.pos[None]
        out = self.embed_norm(out)
        
        for layer in self.main:
            out = layer(out, embed)
        
        out = A.relu(self.ProjNorm(
            tf.einsum('abc,cd->abd',out, self.Proj)
        ))
        
        return tf.reduce_mean(self.final(out), axis=1)
