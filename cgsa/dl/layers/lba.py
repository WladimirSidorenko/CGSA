#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""This module defines a custom neraul-net layer for lexicon-based attention.

"""

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


##################################################################
# Class
class LBA(Layer):
    def __init__(self, lexicon,
                 W_regularizer=None,
                 W_constraint=None,
                 **kwargs):
        """Keras layer that implements a Lexicon-Based Attention mechanism.
        # Input shape 3D tensor with
        shape: `(samples, steps, features)`.
        # Output shape 2D tensor with shape: `(samples, features)`.

        Args:
          lexicon (np.array): lexicon scores of words
          W_regularizer (str or None): regularizer to use for attention matrix
          W_constraint (str or None): constraint to use for attention matrix

        Example:
          model.add(LSTM(64, return_sequences=True))
          model.add(Attention())

        """
        if isinstance(lexicon, dict):
            self.lexicon = np.array(lexicon["value"])
        else:
            self.lexicon = lexicon
        self.W_regularizer = W_regularizer
        self.W_constraint = W_constraint
        self._initial_weights = [self.lexicon]
        super(LBA, self).__init__(**kwargs)

    def build(self, input_shapes):
        lex_indices_shape, rnn_out_shape = input_shapes
        # attention matrix
        self.W = self.add_weight(shape=self.lexicon.shape,
                                 name='{}_W'.format(self.name),
                                 trainable=True,
                                 initializer="Zeros",
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        super(LBA, self).build(input_shapes)

    def call(self, inputs):
        lex_indices, x = inputs
        if K.dtype(lex_indices) != 'int32':
            inputs = K.cast(lex_indices, 'int32')
        lex_scores = K.gather(self.W, lex_indices)
        # sum all scores present in the lexicon
        lex_scores = K.sum(lex_scores, axis=-1)
        # apply non-linerarity
        lex_scores = K.exp(K.tanh(lex_scores))
        # normalize
        lex_scores /= K.cast(K.sum(lex_scores, axis=1, keepdims=True)
                             + K.epsilon(), K.floatx())
        lex_scores = K.expand_dims(lex_scores)
        weighted_input = x * lex_scores
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shapes):
        output_shape = input_shapes[-1]
        return (output_shape[0], output_shape[-1])

    def get_config(self):
        config = super(LBA, self).get_config()
        config.update({
            "lexicon": self.lexicon,
            "W_regularizer": self.W_regularizer,
            "W_constraint": self.W_constraint
        })
        return config
