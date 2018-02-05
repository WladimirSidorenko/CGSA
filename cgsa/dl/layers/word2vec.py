#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from keras.layers.embeddings import Embedding


##################################################################
# Variables and Constants


##################################################################
# Class
class Word2Vec(Embedding):
    """Layer providing pre-trained word2vec embeddings.

    """
    def __init__(self, embs=None, shape=None, trainable=False, **kwargs):
        if embs is None:
            assert shape, ("You should provide either embeddings"
                           " or input shape.")
            self._shape = shape
        else:
            self._shape = embs.shape
            kwargs["weights"] = [embs]

        kwargs["input_dim"] = self._shape[0]
        kwargs["output_dim"] = self._shape[1]
        kwargs["trainable"] = trainable
        super(Word2Vec, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Word2Vec, self).build(input_shape)

    def get_config(self):
        config = super(Word2Vec, self).get_config()
        config.update({'shape': self._shape})
        return config
