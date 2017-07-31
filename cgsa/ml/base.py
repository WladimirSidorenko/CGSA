#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
import abc
import pandas as pd

from cgsa.base import BaseAnalyzer

##################################################################
# Variables and Constants


##################################################################
# Class
class MLBaseAnalyzer(BaseAnalyzer):
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
        super(MLBaseAnalyzer, self).__init__(
            auto_lexicons=auto_lexicons,
            manual_lexicons=manual_lexicons
        )
        self.name = "MLBaseAnalyzer"
        self._model = None
        self.N_JOBS = 1
        self.PARAM_GRID = {}

    def train(self, train_x, train_y, dev_x, dev_y, a_grid_search):
        train_len = len(train_x)
        train_x = [self._extract_feats(t) for t in train_x]
        dev_x = [self._extract_feats(t) for t in dev_x]
        try:
            if a_grid_search:
                cv, train_x, train_y = self._get_cv(
                    train_x, train_y, dev_x, dev_y
                )
                scorer = make_scorer(f1_score, average="micro")
                self._model = GridSearchCV(
                    self._model, self.PARAM_GRID, scoring=scorer,
                    cv=cv, n_jobs=self.N_JOBS, verbose=1
                )
            self._logger.debug("Training classifier %s.", self.name)
            self._model.fit(train_x, train_y)
            self._logger.debug("Classifier %s trained.", self.name)
            if a_grid_search:
                cv_results = pd.DataFrame.from_dict(self._model.cv_results_)
                self._logger.info("CV results:\n%s", cv_results[[
                    "params", "mean_test_score", "std_test_score"
                ]])
                self._logger.info("Best parameters for %s: %r", self.name,
                                  self._model.best_params_)
                self._logger.info("Best score on held-out set: %r",
                                  self._model.best_score_)
        except:
            # `dev_x' and `dev_y' might have been appended to `train_x' and
            # `train_y' while generating the CV set
            train_x = train_x[:train_len]
            train_y = train_y[:train_len]
            raise

    def predict(self, msg):
        # no training is required for this method
        raise NotImplementedError

    @abc.abstractmethod
    def _extract_feats(self, a_tweet):
        """Method for training the model.

        Args:
          a_tweet (cgsa.data.Tweet):
            training instance to extract features from

        Returns:
          dict: extracted features and their values

        """
        raise NotImplementedError
