#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from keras import backend as K
from keras import activations, constraints, initializers, regularizers
from keras.engine import InputSpec
from keras.layers.recurrent import Recurrent

from .common import DFLT_INITIALIZER
from .utils import dot_product, get_subtensor, set_subtensor


##################################################################
# Variables and Constants

##################################################################
# Class
class RN(Recurrent):
    def __init__(self,
                 activation='tanh',
                 use_bias=False,
                 recurrent_initializer=DFLT_INITIALIZER,
                 bias_initializer=DFLT_INITIALIZER,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 recurrent_constraint=None,
                 bias_constraint=None, **kwargs):
        """Recursive neural layer.

        """
        kwargs["name"] = kwargs.get("name", "RN")
        super(RN, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3)]
        self.state_spec = [InputSpec(ndim=3)]
        self.activation = activations.get(activation)

        self.W_initializer = initializers.get(recurrent_initializer)
        self.b_initializer = initializers.get(bias_initializer)

        self.W_regularizer = regularizers.get(recurrent_regularizer)
        self.b_regularizer = regularizers.get(bias_regularizer)

        self.W_constraint = constraints.get(recurrent_constraint)
        self.b_constraint = constraints.get(bias_constraint)
        self.use_bias = use_bias

    def __call__(self, inputs, **kwargs):
        assert len(inputs) > 1, (
            "The input to the {:s} layer should include"
            " embeddings as the second argument").format(self.name)
        self.EMBS = inputs[1]
        self.units = K.int_shape(self.EMBS)[-1]
        return super(RN, self).__call__(inputs, **kwargs)

    def build(self, input_shapes):
        # In Socher's RNNs the dimensionality of the output always matches that
        # of the input.
        self.states = [None]
        self.state_spec = [InputSpec(shape=input_shapes[1])]
        self.W = self.add_weight(shape=(self.units * 2, self.units),
                                 name="{}_W".format(self.name),
                                 initializer=self.W_initializer,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.use_bias:
            self.b = self.add_weight(shape=(self.units,),
                                     name="{}_b".format(self.name),
                                     initializer=self.b_initializer,
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.bias = None
        super(RN, self).build(input_shapes)

    def preprocess_input(self, inputs, training=None):
        return inputs

    def get_initial_state(self, inputs):
        return [inputs[1]]

    def step(self, inputs, states):
        inst_indcs = K.arange(0, K.shape(inputs)[0])
        chld_indcs = inputs[:, 0]
        prnt_indcs = inputs[:, 1]
        embs = states[0]
        chld_embs = get_subtensor(embs, inst_indcs, chld_indcs)
        prnt_embs = get_subtensor(embs, inst_indcs, prnt_indcs)
        ret = dot_product(K.concatenate([chld_embs, prnt_embs], axis=-1),
                          self.W)
        if self.use_bias:
            ret = K.bias_add(ret, self.b)
        ret = self.activation(ret)
        # now, the tricky part we need to actually modify the embedding matrix
        embs = set_subtensor(embs, ret, inst_indcs, prnt_indcs)
        return ret, [embs]

    def get_config(self):
        config = super(RN, self).get_config()
        config.update({
            "activation": activations.serialize(self.activation),
            "use_bias": self.use_bias,
            "recurrent_initializer": initializers.serialize(
                self.W_initializer),
            "bias_initializer": initializers.serialize(
                self.b_initializer
            ),
            "recurrent_regularizer": regularizers.serialize(
                self.W_regularizer
            ),
            "bias_regularizer": regularizers.serialize(
                self.b_regularizer
            ),
            "recurrent_constraint": constraints.serialize(
                self.W_constraint
            ),
            "bias_constraint": constraints.serialize(
                self.b_constraint
            )
        })
        return config
