#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""This module defines a custom neraul-net layer for context-based attention.

"""

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from keras import backend as K
from keras.engine.topology import Layer


from .utils import get_subtensor


##################################################################
# Class
class CBA(Layer):
    def __init__(self,
                 W_regularizer=None,
                 W_constraint=None,
                 **kwargs):
        """Keras layer that implements a Context-Based Attention mechanism.

        Args:
          W_regularizer (str or None): regularizer to use for attention matrix
          W_constraint (str or None): constraint to use for attention matrix

        See example in `cgsa/dl/lba.py`.

        """
        self.W_regularizer = W_regularizer
        self.W_constraint = W_constraint
        super(CBA, self).__init__(**kwargs)

    def build(self, input_shapes):
        emb_shape, _, _, rnn_shape = input_shapes
        emb_dim = emb_shape[-1]
        rnn_dim = rnn_shape[-1]
        # attention matrix
        self.W = self.add_weight(shape=(emb_dim + rnn_dim, rnn_dim),
                                 name='{}_W'.format(self.name),
                                 trainable=True,
                                 initializer="Orthogonal",
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        super(CBA, self).build(input_shapes)

    def call(self, inputs):
        embs, prnt_indices, lba_out, rnn_out = inputs
        n_istances = K.shape(embs)[0]
        instance_len = K.shape(embs)[1]
        inst_indcs = K.arange(0, n_istances)
        inst_indcs = K.repeat_elements(inst_indcs, instance_len, axis=0)
        # obtain LBA outut of the parent nodes
        prnt_lba = get_subtensor(lba_out, inst_indcs, K.flatten(prnt_indices))
        prnt_lba = K.reshape(prnt_lba, rnn_out.shape)
        # concatenate input embeddings with the output of LBA
        cba_input = K.concatenate([prnt_lba, embs], axis=-1)
        cba_output = K.sum(K.dot(cba_input, self.W), axis=-1)
        cba_output = K.exp(K.tanh(cba_output))
        cba_output /= K.cast(K.sum(cba_output, axis=1, keepdims=True)
                             + K.epsilon(), K.floatx())
        cba_output = K.expand_dims(cba_output)
        weighted_output = rnn_out * cba_output
        return K.sum(weighted_output, axis=1)

    def compute_output_shape(self, input_shapes):
        output_shape = input_shapes[-1]
        return (output_shape[0], output_shape[-1])

    def get_config(self):
        config = super(CBA, self).get_config()
        config.update({
            "W_regularizer": self.W_regularizer,
            "W_constraint": self.W_constraint
        })
        return config
