#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from keras.models import Sequential
from keras.layers import (BatchNormalization, Conv1D, Dense,
                          GlobalMaxPooling1D)
from cgsa.dl.base import DLBaseAnalyzer

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
        self.name = "Severyn"
        self._min_wdth = 5
        self._flt_wdth = 5
        self._n_filters = 300

    def _init_nn(self):
        self.init_w_emb()
        self._model = Sequential()
        # add embedding layer
        self._model.add(self.W_EMB)
        self._model.add(BatchNormalization())
        # add convolutional layer
        self._model.add(Conv1D(
            self._n_filters, self._flt_wdth,
            activation="relu", kernel_initializer="he_normal"))
        # add max-pooling
        self._model.add(GlobalMaxPooling1D())
        self._model.add(BatchNormalization())
        # add final dense layer
        self._model.add(Dense(self._n_y,
                              activation="softmax",
                              kernel_initializer="he_normal",
                              bias_initializer="he_normal"))
        self._model.compile(optimizer="rmsprop",
                            metrics=["accuracy"],
                            loss="categorical_hinge")
