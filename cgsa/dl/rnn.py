#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from .tree_rnn_base import TreeRNNBaseAnalyzer
from .layers import RN

##################################################################
# Variables and Constants


##################################################################
# Class
class RNNAnalyzer(TreeRNNBaseAnalyzer):
    """Class for DL-based sentiment analysis.

    Attributes:

    """
    def __init__(self, *args, **kwargs):
        super(RNNAnalyzer, self).__init__(*args, **kwargs)
        self.name = "rnn"
        # self.ndim = 35

    def _init_rnn(self, inputs):
        """Method defining a recurrent unit.

        Returns:
          keras.layer: instanse of a custom keras layer

        """
        return RN()(inputs)
