#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from cgsa.base import BaseAnalyzer

##################################################################
# Variables and Constants


##################################################################
# Class
class SeverynAnalyzer(BaseAnalyzer):
    """Class for DL-based sentiment analysis.

    Attributes:

    """

    def __init__(self, lexicons=[]):
        """Class constructor.

        Args:
          lexicons (list[str]): list of lexicons to use for prediction

        """
        self._read_lexicons(lexicons)

    def train(self, train_data, dev_data=None):
        # no training is required for this method
        raise NotImplementedError

    def predict(self, msg):
        # no training is required for this method
        raise NotImplementedError
