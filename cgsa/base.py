#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing abstract base for all sentiment analyzers.

Attributes:
  BaseAnalyzer (class): abstract base for all sentiment analyzers

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

import abc

##################################################################
# Variables and Constants


##################################################################
# Classes
class BaseAnalyzer(object):
    """Abstract class for coarse-grained sentiment analyzers.

    Attributes:

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """Class constructor.

        """
        pass

    @abc.abstractmethod
    def train(self, train_data, dev_data=None):
        """Method for training the model.

        Args:
          train_data (2-tuple(list, dict)):
            list of gold relations and dict with parses
          dev_data (2-tuple(list, dict) or None):
            list of development relations and dict with parses

        Returns:
          void:

        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, a_rel, a_data, a_ret, a_i):
        """Method for predicting sense of single relation.

        Args:
          a_rel (dict):
            discourse relation whose sense should be predicted
          a_data (2-tuple(dict, dict)):
            list of input JSON data
          a_ret (np.array):
            output prediction vector
          a_i (int):
            row index in the output vector

        Returns:
          void:

        Note:
          updates ``a_ret`` in place

        """
        raise NotImplementedError
