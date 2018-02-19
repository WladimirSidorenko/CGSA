#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from keras.models import Sequential
from keras.layers import (Dense, Dropout, Bidirectional, LSTM,
                          GaussianNoise)
from keras.regularizers import l2

from .base import DLBaseAnalyzer, L2_COEFF
from .layers import Attention


##################################################################
# Variables and Constants


##################################################################
# Class
class BaziotisAnalyzer(DLBaseAnalyzer):
    """Class for DL-based sentiment analysis.

    Attributes:

    """
    def __init__(self, *args, **kwargs):
        super(BaziotisAnalyzer, self).__init__(*args, **kwargs)
        self.name = "baziotis"
        # Throughout this classifier, we use the same hyper-parameters as the
        # ones used by Baziotis et al. in their original work.
        # self.ndim = 300

    def _init_nn(self):
        self.init_w_emb()
        self._model = Sequential()
        # add embedding layer
        self._model.add(self.W_EMB)
        self._model.add(GaussianNoise(0.2))
        self._model.add(Dropout(0.3))
        # add two BiLSTM layers
        for _ in range(2):
            self._model.add(
                Bidirectional(
                    LSTM(150, recurrent_dropout=0.25,
                         return_sequences=True,
                         activity_regularizer=l2(L2_COEFF),
                         kernel_regularizer=l2(L2_COEFF),
                         bias_regularizer=l2(L2_COEFF),
                         recurrent_regularizer=l2(L2_COEFF))
                )
            )
            self._model.add(Dropout(0.5))
        # add Attention layer
        self._model.add(Attention(bias=False))
        self._model.add(Dropout(0.5))
        # add the final dense layer
        self._model.add(Dense(self._n_y,
                              activation="softmax",
                              activity_regularizer=l2(L2_COEFF),
                              kernel_regularizer=l2(L2_COEFF),
                              bias_regularizer=l2(L2_COEFF)))
        self._model.compile(optimizer="adadelta",
                            metrics=["categorical_accuracy"],
                            loss="categorical_hinge")
