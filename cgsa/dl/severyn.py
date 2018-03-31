#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from keras.models import Sequential
from keras.layers import (Conv1D, Dense, Dropout, GlobalMaxPooling1D)
from keras.regularizers import l2

from .base import DFLT_TRAIN_PARAMS, DLBaseAnalyzer, L2_COEFF
from .layers import DFLT_INITIALIZER


##################################################################
# Variables and Constants


##################################################################
# Class
class SeverynAnalyzer(DLBaseAnalyzer):
    """Class for DL-based sentiment analysis.

    Attributes:

    """
    # it's actually filter width

    def __init__(self, *args, **kwargs):
        super(SeverynAnalyzer, self).__init__(*args, **kwargs)
        self.name = "severyn"
        # The default dimensionality of word embeddings d is set to 100 (might
        # be overwitten by the loaded word2vec vectors).
        # self.ndim = 100
        # The width m of the convolution filters is set to 5 and the number of
        # convolutional feature maps is 300.
        self._min_width = 5
        self._flt_width = 5
        self._n_filters = 300

    def _init_nn(self):
        self.init_w_emb()
        self._model = Sequential()
        # add embedding layer
        self._model.add(self.W_EMB)
        # add convolutional layer
        self._model.add(Conv1D(
            self._n_filters, self._flt_width,
            activation="relu",
            kernel_initializer=DFLT_INITIALIZER,
            kernel_regularizer=l2(L2_COEFF),
            bias_regularizer=l2(L2_COEFF)))
        # add max-pooling
        self._model.add(GlobalMaxPooling1D())
        # add dropout as penultimate layer
        self._model.add(Dropout(0.5))
        # add final dense layer
        self._model.add(Dense(self._n_y,
                              activation="softmax",
                              kernel_initializer=DFLT_INITIALIZER,
                              bias_initializer=DFLT_INITIALIZER,
                              kernel_regularizer=l2(L2_COEFF),
                              bias_regularizer=l2(L2_COEFF)))
        self._model.compile(**DFLT_TRAIN_PARAMS)
        self._logger.debug(self._model.summary())
