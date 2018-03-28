#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from keras import backend as K
from keras.engine import InputSpec
from keras.layers.recurrent import Recurrent

##################################################################
# Variables and Constants


##################################################################
# Class
class MSRN(Recurrent):
    def __init__(self, *args, **kwargs):
        super(MSRN, self).__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim=4)]
        self.state_spec = [InputSpec(ndim=4)]

    def __call__(self, inputs, **kwargs):
        self.units = K.int_shape(inputs)[-1]
        return super(MSRN, self).__call__(inputs, **kwargs)

    def build(self, input_shapes):
        super(MSRN, self).build([input_shapes])
        self.states = [None]
        self._eye = K.eye(input_shapes[-1])

    def compute_output_shape(self, input_shape):
        output_shape = super(MSRN, self).compute_output_shape(input_shape)
        return output_shape + (self.units,)

    def get_initial_state(self, inputs):
        istate = K.zeros_like(inputs)
        istate = K.sum(istate, axis=(1,)) + self._eye
        return [istate]

    def step(self, inputs, states):
        ret = K.batch_dot(inputs, states[0], axes=[2, 1])
        return ret, [ret]
