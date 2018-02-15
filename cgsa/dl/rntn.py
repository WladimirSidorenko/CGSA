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
from cgsa.dl.base import DLBaseAnalyzer

from .tree_rnn_base import TreeRNNBaseAnalyzer

##################################################################
# Variables and Constants


##################################################################
# Class
class RNTNAnalyzer(TreeRNNBaseAnalyzer):
    """Class for DL-based sentiment analysis.

    Attributes:

    """
    def __init__(self, *args, **kwargs):
        super(RNTNAnalyzer, self).__init__(*args, **kwargs)
        self.name = "rntn"
        # default dimensionality for task-specific vectors
        self.ndim = 300
