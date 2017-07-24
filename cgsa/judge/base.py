#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for meta-classification.

Attributes:
  BaseJudge (class): class for joining decisions of single classifiers

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

import abc

##################################################################
# Variables and Constants


##################################################################
# Class
class BaseJudge(object):
    """Meta-classifier.

    This classifier unites decisions of other multiple independent classifiers.

    Attrs:

    Methods:

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, a_n_x, a_n_y):
        """Class constructor.

        Args:
        a_n_x (int):
          number of underlying cassifiers
        a_n_y (int):
          number of classes to predict


        """
        self.n_x = a_n_x
        self.n_y = a_n_y

    @abc.abstractmethod
    def train(self, a_ts, a_dev_data=None):
        """Method for training the model.

        Args:
        a_ts (list(2-tuple(x, y))):
          list of training JSON data
        a_dev_data (2-tuple(dict, dict) or None):
          list of development JSON data

        Returns:
        (void)

        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, a_x):
        """Method for predicting sense of single relation.

        Args:
        a_x (np.array):
          (submodels x class) array of input predictions

        Returns:
        str:
          most probable sense of discourse relation

        """
        raise NotImplementedError
