#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from six.moves import xrange
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from cgsa.ml.base import MLBaseAnalyzer

##################################################################
# Variables and Constants
DFLT_C = 0.02
DFLT_CLS_WGHT = None
DFLT_PARAMS = {"class_weight": DFLT_CLS_WGHT, "loss": "hinge",
               "penalty": "l1", "dual": True,
               "multi_class": "crammer_singer"}
PARAM_GRID = {"clf__C": [float(i)/100. for i in xrange(1, 3)]}


##################################################################
# Class
class MohammadAnalyzer(MLBaseAnalyzer):
    """Class for ML-based sentiment analysis.

    Attributes:

    """

    def __init__(self, auto_lexicons=[], manual_lexicons=[], a_clf=None):
        """Class constructor.

        Args:
          auto_lexicons (list[str]): list of automatically compiled
            lexicons
          manual_lexicons (list[str]): list of manually compiled lexicons
          a_clf (None or Classifier Instance): classifier to use (None for
            default)

        """
        super(MohammadAnalyzer, self).__init__()
        self.name = "NRC-Canada"
        clf = a_clf or LinearSVC(C=DFLT_C, **DFLT_PARAMS)
        self._model = Pipeline([("vect", DictVectorizer()),
                                ("clf", clf)])
        self.N_JOBS = 1
        self.PARAM_GRID = PARAM_GRID

    def predict(self, msg):
        # no training is required for this method
        raise NotImplementedError

    def _extract_feats(self, a_tweet):
        """Method for training the model.

        Args:
          a_tweet (cgsa.data.Tweet):
            training instance to extract features from

        Returns:
          dict: extracted features and their values

        """
        return {"hui": 1}
