#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from keras.models import Sequential
from keras.layers import (Dense, LSTM)
from keras.layers.wrappers import Bidirectional
from .base import DFLT_TRAIN_PARAMS, DLBaseAnalyzer
from .layers import DFLT_INITIALIZER

##################################################################
# Variables and Constants


##################################################################
# Class
class BiLSTMAnalyzer(DLBaseAnalyzer):
    """Class for DL-based sentiment analysis.

    Attributes:

    """
    # it's actually filter width

    def __init__(self, *args, **kwargs):
        super(BiLSTMAnalyzer, self).__init__(*args, **kwargs)
        self.name = "BiLSTM"
        self._n_epochs = 5

    def _init_nn(self):
        self.init_w_emb()
        self._model = Sequential()
        # add embedding layer
        self._model.add(self.W_EMB)
        # add recurrent layers
        self._model.add(Bidirectional(
            LSTM(16, dropout=0.2, return_sequences=True)))
        self._model.add(
            LSTM(16, dropout=0.2))
        # add final dense layer
        self._model.add(Dense(self._n_y,
                              activation="softmax",
                              kernel_initializer=DFLT_INITIALIZER,
                              bias_initializer=DFLT_INITIALIZER))
        self._model.compile(**DFLT_TRAIN_PARAMS)
