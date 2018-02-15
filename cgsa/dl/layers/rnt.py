#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from keras import backend as K
from keras.layers.recurrent import Recurrent

##################################################################
# Variables and Constants


##################################################################
# Class
class RNT(Recurrent):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, initializer="glorot_uniform", **kwargs):
        """Recursive neural tensor layer.

        """
        self.supports_masking = True
        self.initializer = initializer

        self.W_regularizer = W_regularizer
        self.b_regularizer = b_regularizer

        self.W_constraint = W_constraint
        self.b_constraint = b_constraint

        self.bias = bias
        super(RNT, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.initializer,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            if input_shape[1] is None:
                raise RuntimeError(
                    "Cannot initialize bias term {!r} with non-fixed input length.".format(self.bias)
                )
            self.b = self.add_weight(shape=(input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

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
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if output_shape[-1] is not None:
            output_shape[-1] = 1
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = {
            "initializer": self.initializer,
            "W_regularizer": self.W_regularizer,
            "b_regularizer": self.b_regularizer,
            "W_constraint": self.W_constraint,
            "b_constraint": self.b_constraint,
            "bias": self.bias
        }
        return config
