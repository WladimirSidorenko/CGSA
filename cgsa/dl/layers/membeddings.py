#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from keras.layers.embeddings import Embedding
import keras.backend as K


##################################################################
# Variables and Constants
EMBEDDING_LAYER_NAME = "MatrixEmbeddings"


##################################################################
# Methods
def diagonalize(inputs, eye):
    """Convert batch of vectors to diagonal matrices.

    Args:
      inputs (tensor): batch of vectors of size ()

    Returns:
      tensor: input vectors as diagonal matrices

    """
    return K.tile


##################################################################
# Class
class MatrixEmbedding(Embedding):
    """Layer providing matrix embeddings of words.

    """

    def build(self, input_shape):
        self.U = self.add_weight(
            shape=(self.input_dim, 1, self.output_dim),
            initializer=self.embeddings_initializer,
            name='U_embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype)
        self.V = self.add_weight(
            shape=(self.input_dim, self.output_dim, 1),
            initializer=self.embeddings_initializer,
            name='V_embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype)
        self.DIAG = self.add_weight(
            shape=(self.input_dim, 1, self.output_dim),
            initializer=self.embeddings_initializer,
            name='DIAG_embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype)
        self._eye = K.eye(self.output_dim)
        self.built = True

    def compute_output_shape(self, input_shape):
        output_shape = super(MatrixEmbedding, self).compute_output_shape(
            input_shape
        )
        return output_shape + (self.output_dim,)

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        # U will have shape `batch_size x output_dim'
        U = K.tile(K.gather(self.U, inputs), (1, 1, self.output_dim, 1))
        # V will have shape `batch_size x output_dim'
        V = K.tile(K.gather(self.V, inputs), (1, 1, 1, self.output_dim))
        DIAG = K.tile(K.gather(self.DIAG, inputs), (1, 1, self.output_dim, 1))
        ret = U * V + DIAG * self._eye
        return ret
