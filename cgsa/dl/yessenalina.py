#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from keras.layers import Dense, Flatten, Input
from keras.models import Model
from keras.regularizers import l2

from .base import EMB_INDICES_NAME, DLBaseAnalyzer, L2_COEFF
from .layers import MSRN, YMatrixEmbedding

##################################################################
# Variables and Constants


##################################################################
# Class
class YessenalinaAnalyzer(DLBaseAnalyzer):
    """Class for DL-based sentiment analysis.

    Attributes:

    """
    def __init__(self, *args, **kwargs):
        super(YessenalinaAnalyzer, self).__init__(*args, **kwargs)
        if self._w2v or self._lstsq:
            raise NotImplementedError(
                "w2v and lest square vectors are not"
                " supported by this type of the model.")
        self.name = "yessenalina"
        # according to the original Yessenalina's implementation
        # self.ndim = 3

    def _init_rnn(self, inputs):
        """Method defining a recurrent unit.

        Returns:
          keras.layer: instanse of a custom keras layer

        """
        return MSRN()(inputs)

    def _init_nn(self):
        """Initialize neural network.

        """
        self.init_w_emb()
        emb_indices = Input(shape=(None,),
                            dtype="int32",
                            name=EMB_INDICES_NAME)
        mtx_embs = self.MTX_EMB(emb_indices)
        rnn = self._init_rnn(mtx_embs)
        flat = Flatten()(rnn)
        out = Dense(self._n_y,
                    activation="softmax",
                    kernel_regularizer=l2(L2_COEFF),
                    bias_regularizer=l2(L2_COEFF),
                    name="dense")(flat)
        self._model = Model(inputs=[emb_indices], outputs=out)
        self._model.compile(optimizer="adadelta",
                            metrics=["categorical_accuracy"],
                            loss="categorical_hinge")
        self._logger.debug(self._model.summary())

    def _init_w_emb(self):
        """Initialize matrix embeddings along with vector representations.

        """
        self.MTX_EMB = YMatrixEmbedding(len(self._w2i), self.ndim,
                                        embeddings_initializer="he_normal",
                                        embeddings_regularizer=l2(L2_COEFF))
