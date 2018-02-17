#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from keras import backend as K
from keras.engine import InputSpec
from .rn import RN, set_subtensor

##################################################################
# Variables and Constants


##################################################################
# Class
class MVRN(RN):
    def __init__(self, *args, **kwargs):
        super(MVRN, self).__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3),
                           InputSpec(ndim=4)]

    def build(self, input_shapes):
        super(MVRN, self).build(input_shapes)
        self.states = [None, None]
        self.state_spec = [InputSpec(shape=input_shapes[1]),
                           InputSpec(shape=input_shapes[2])]
        self.W_m = self.add_weight(shape=(2 * self.units, self.units),
                                   name="{}_W_m".format(self.name),
                                   initializer=self.W_initializer,
                                   regularizer=self.W_regularizer,
                                   constraint=self.W_constraint)

    def get_initial_state(self, inputs):
        return [inputs[1], inputs[2]]

    def step(self, inputs, states):
        inst_indcs = K.arange(0, K.shape(inputs)[0])
        chld_indcs = inputs[:, 0]
        prnt_indcs = inputs[:, 1]

        # vector representation of words
        emb_vecs = states[0]
        # matrix representation of words
        emb_mtcs = states[1]

        chld_vecs = emb_vecs[inst_indcs, chld_indcs]
        chld_mtcs = emb_mtcs[inst_indcs, chld_indcs]
        prnt_vecs = emb_vecs[inst_indcs, prnt_indcs]
        prnt_mtcs = emb_mtcs[inst_indcs, prnt_indcs]
        # multiply child vectors with parent matrices and vice versa
        chld_vecs = K.batch_dot(chld_vecs, prnt_mtcs)
        prnt_vecs = K.batch_dot(prnt_vecs, chld_mtcs)
        # compute new vector representations: `vec_ret` will have the
        # dimension: `batch_size x units`
        vec_ret = K.dot(K.concatenate([chld_vecs, prnt_vecs], axis=-1),
                        self.W)
        if self.use_bias:
            vec_ret = K.bias_add(vec_ret, self.b)
        vec_ret = self.activation(vec_ret)
        emb_vecs = set_subtensor(emb_vecs[inst_indcs, prnt_indcs], vec_ret)

        # `node_matrices` will have the shape `batch_size x units x 2 * units`
        node_matrices = K.concatenate([chld_mtcs, prnt_mtcs], axis=-1)
        # compute new matrix representations: `mtx_ret` will have the shape
        # `batch_size x units x units`
        mtx_ret = K.dot(node_matrices, self.W_m)
        emb_mtcs = set_subtensor(emb_mtcs[inst_indcs, prnt_indcs],
                                 mtx_ret)
        # return newly computed vector representations, and updated vectors and
        # matrices
        return vec_ret, [emb_vecs, emb_mtcs]
