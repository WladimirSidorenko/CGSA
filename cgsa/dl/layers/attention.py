#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""This module was copied in parts from the [`keras_utilities`
library](https://github.com/cbaziotis/keras-utilities) and updated to the
latest Keras interface.

The license of the `keras

MIT License

Copyright (c) 2017 Christos Baziotis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from keras import backend as K
from keras.engine.topology import Layer

##################################################################
# Variables and Constants


##################################################################
# Class
class RawAttention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=False, initializer="glorot_uniform", **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with
          return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        self.initializer = initializer

        self.W_regularizer = W_regularizer
        self.b_regularizer = b_regularizer

        self.W_constraint = W_constraint
        self.b_constraint = b_constraint

        self.bias = bias
        super(RawAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            if input_shape[1] is None:
                raise ValueError(
                    ("Cannot initialize bias term {!r}"
                     " with non-fixed input length.").format(self.bias)
                )
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None
        super(RawAttention, self).build(input_shape)

    def call(self, x, mask=None):
        eij = K.dot(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may
        # be almost zero and this results in NaN's. A workaround is to add a
        # very small positive number ε to the sum.  a /= K.cast(K.sum(a,
        # axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        return a

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], 1)

    def get_config(self):
        config = super(RawAttention, self).get_config()
        config.update({
            "initializer": self.initializer,
            "W_regularizer": self.W_regularizer,
            "b_regularizer": self.b_regularizer,
            "W_constraint": self.W_constraint,
            "b_constraint": self.b_constraint,
            "bias": self.bias
        })
        return config


class MergeAttention(Layer):
    def __init__(self, **kwargs):
        super(MergeAttention, self).__init__(**kwargs)

    def call(self, inputs):
        x, attention = inputs
        return self._merge(x, attention)

    def compute_output_shape(self, input_shapes):
        output_shape = input_shapes[0]
        return ((output_shape[0], output_shape[-1]))

    def _merge(self, x, attention):
        x = x * attention
        return K.sum(x, axis=1)


class Attention(RawAttention, MergeAttention):
    def call(self, x, **kwargs):
        attention = RawAttention.call(self, x, **kwargs)
        return MergeAttention.call(self, [x, attention])

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
