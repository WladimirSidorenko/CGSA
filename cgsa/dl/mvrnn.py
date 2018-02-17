#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from keras.layers import Dense, Input
from keras.models import Model
from keras.regularizers import l2

from .base import L2_COEFF
from .tree_rnn_base import (TreeRNNBaseAnalyzer, DEP_LAYER_NAME,
                            EMB_INDICES_NAME)
from .layers import MVRN
from .layers import MatrixEmbedding

##################################################################
# Variables and Constants


##################################################################
# Class
class MVRNNAnalyzer(TreeRNNBaseAnalyzer):
    """Class for DL-based sentiment analysis.

    Attributes:

    """
    def __init__(self, *args, **kwargs):
        super(MVRNNAnalyzer, self).__init__(*args, **kwargs)
        if self._w2v or self._lstsq:
            raise NotImplementedError(
                "w2v and lest square vectors are not"
                " supported by this type of the model.")

        self.name = "mvrnn"
        # according to the original Socher implementation
        self.ndim = 50
        self.r = 3

    def _init_rnn(self, inputs):
        """Method defining a recurrent unit.

        Returns:
          keras.layer: instanse of a custom keras layer

        """
        return MVRN()(inputs)

    def _init_nn(self):
        """Initialize neural network.

        """
        self.init_w_emb()
        dependencies = Input(shape=(None, 2),
                             dtype="int32",
                             name=DEP_LAYER_NAME)
        emb_indices = Input(shape=(None,),
                            dtype="int32",
                            name=EMB_INDICES_NAME)
        vec_embs = self.W_EMB(emb_indices)
        mtx_embs = self.MTX_EMB(emb_indices)
        rnn = self._init_rnn([dependencies, vec_embs, mtx_embs])
        out = Dense(self._n_y,
                    activation="softmax",
                    kernel_regularizer=l2(L2_COEFF),
                    bias_regularizer=l2(L2_COEFF),
                    name="dense")(rnn)
        self._model = Model(inputs=[dependencies, emb_indices],
                            outputs=out)
        self._model.compile(optimizer="adadelta",
                            metrics=["categorical_accuracy"],
                            loss="categorical_hinge")
        self._logger.debug(self._model.summary())

    def _init_w_emb(self):
        """Initialize matrix embeddings along with vector representations.

        """
        super(MVRNNAnalyzer, self)._init_w_emb()
        self.MTX_EMB = MatrixEmbedding(len(self._w2i), self.ndim,
                                       embeddings_initializer="he_normal",
                                       embeddings_regularizer=l2(L2_COEFF))
