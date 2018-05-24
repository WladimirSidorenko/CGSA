#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from keras import backend as K
from .rn import RN
from .utils import dot_product, get_subtensor, set_subtensor

##################################################################
# Variables and Constants


##################################################################
# Class
class RNT(RN):
    def build(self, input_shapes):
        self.V = self.add_weight(shape=(self.units, 2 * self.units,
                                        2 * self.units),
                                 name="{}_V".format(self.name),
                                 initializer=self.W_initializer,
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        super(RNT, self).build(input_shapes)

    def step(self, inputs, states):
        inst_indcs = K.arange(0, K.shape(inputs)[0])
        chld_indcs = inputs[:, 0]
        prnt_indcs = inputs[:, 1]
        embs = states[0]
        chld_embs = get_subtensor(embs, inst_indcs, chld_indcs)
        prnt_embs = get_subtensor(embs, inst_indcs, prnt_indcs)
        node_embs = K.concatenate([chld_embs, prnt_embs], axis=-1)
        node_embs_t = K.transpose(node_embs)

        # node_embs_t will have the dimension (2 * units x batch_size)
        # V will have the dimension (units x 2 * units x 2 * units)
        # ret0 will have the dimension (units x 2 * units x batch_size)
        ret0 = dot_product(self.V, node_embs_t)
        # ret0 will have the dimension (units x 2 * units x batch_size)
        # tiled node embs will have the dimension (units x 2 * units x batch_size)
        # ret1 will have the dimension (units x batch_size x 2 * units)
        ret1 = ret0 * K.tile(node_embs_t, (self.units, 1, 1))
        # ret2 will have the dimension (batch_size x units)
        ret2 = K.transpose(K.sum(ret1, axis=1))
        # ret will have the same dimensions as ret2
        ret = ret2 + dot_product(node_embs, self.W)
        if self.use_bias:
            ret = K.bias_add(ret, self.b)
        ret = self.activation(ret)
        # now, the tricky part we need to actually modify the embedding matrix
        embs = set_subtensor(embs, ret, inst_indcs, prnt_indcs)
        return ret, [embs]
