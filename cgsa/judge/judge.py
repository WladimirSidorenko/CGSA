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
class SumJudge(object):
    """Meta-classifier for weighting decisions of multiple classifiers.

    Attrs:
      train(): method for training meta-classifiers
      test(): method for joint predictions

    """

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

    def predict(self, scores):
        """Method for predicting sense of single relation.

        Args:
          scores (np.array):
            array of classifiers' predictions

        Returns:
          2-tuple: index of the most likely class and the total score vector

        """
        ret = np.sum(scores, axis=0)
        return (np.argmax(ret), ret)


class DefaultJudge(SumJudge):
    pass
