#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from keras.layers.embeddings import Embedding
import keras.backend as K
import numpy as np

from .rn import set_subtensor


##################################################################
# Variables and Constants
EMBEDDING_LAYER_NAME = "YMatrixEmbeddings"


##################################################################
# Class
class YMatrixEmbedding(Embedding):
    """Layer providing matrix embeddings of words.

    """

    def build(self, input_shape):
        self.EMBS = self.add_weight(
            shape=(self.input_dim,
                   self.output_dim,
                   self.output_dim),
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype)
        self.zero_one_vec = np.zeros(self.output_dim, dtype="float32")
        self.zero_one_vec[-1] = 1.
        self.built = True

    def compute_output_shape(self, input_shape):
        output_shape = super(YMatrixEmbedding, self).compute_output_shape(
            input_shape
        )
        return output_shape + (self.output_dim,)

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        # embs will have shape `batch_size x instance length x output_dim x
        # output_dim'
        embs = K.gather(self.EMBS, inputs)
        # return set_subtensor(embs[:, :, -1], self.zero_one_vec)
        return embs
