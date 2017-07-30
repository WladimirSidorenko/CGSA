#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing a class to judge classification.

Attributes:
  Judge (class): class for merging judgments of different classifiers

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

import numpy as np


##################################################################
# Classes
class Judge(object):
    """Meta-classifier for weighting decisions of multiple classifiers.

    Attrs:
      __init__(): class constructor
      train(): method for training meta-classifiers
      test(): method for joint predictions

    """

    def __init__(self, a_n_x, a_n_y):
        """Class constructor.

        Args:
        a_n_x (int):
          number of underlying cassifiers
        a_n_y (int):
          number of classes to predict


        """
        pass

    def train(self, a_train, a_dev=()):
        """Method for training the model.

        Args:
        a_train (list(3-tuple(x, rel, y))):
          list of training instances
        a_dev (2-tuple(dict, dict) or None):
          list of development instances

        Returns:
        (void)

        """
        return

    def predict(self, a_rel, a_x):
        """Method for predicting sense of single relation.

        Args:
        a_rel (dict):
          input relation to classify
        a_x (np.array):
          (submodels x class) array of input predictions

        Returns:
        str:
          most probable sense of discourse relation

        """
        ret = np.mean(a_x, axis=0)
        return (np.argmax(ret), ret)
        if is_explicit(a_rel):
            return self.explicit.predict(a_x)
        return self.implicit.predict(a_x)

    def _free(self):
        """Free resources used by the model.

        Args:
        (void):

        Returns:
        (void):

        """
        del self.explicit
        del self.implicit

    def _divide_data(self, a_data):
        """Separate dataset into explicit and implicit instances.

        Args:
        a_data (2-tuple(dict, dict)):
        list of gold relations and dict with parses

        Returns:
        (2-tuple(list, list)):
          lists of explicit and implicit training instances

        """
        if a_data is None:
            return ((), ())
        explicit_instances = []
        implicit_instances = []
        for x_i, irel, y_i in a_data:
            if is_explicit(irel):
                explicit_instances.append((x_i, y_i))
            else:
                implicit_instances.append((x_i, y_i))
        return (explicit_instances, implicit_instances)
