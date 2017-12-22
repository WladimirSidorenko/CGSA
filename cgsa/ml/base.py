#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from collections import defaultdict
from six import iteritems, string_types
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
import abc
import numpy as np
import pandas as pd

from cgsa.base import BaseAnalyzer
from cgsa.utils.common import LOGGER

##################################################################
# Variables and Constants
ALEX = "alex_"


##################################################################
# Class
class MLBaseAnalyzer(BaseAnalyzer):
    """Class for ML-based sentiment analysis.

    Attributes:

    """

    def __init__(self, **kwargs):
        """Class constructor.

        Args:
          kwargs (dict): additional keyword arguments

        """
        super(MLBaseAnalyzer, self).__init__(**kwargs)
        self.name = "MLBaseAnalyzer"
        self._model = None
        self.N_JOBS = 1
        self.PARAM_GRID = {}

    def restore(self):
        """Restore members which could not be serialized.

        """
        self._logger = LOGGER

    def train(self, train_x, train_y, dev_x, dev_y, a_grid_search):
        self._logger.debug("Training %s...", self.name)
        train_len = len(train_x)
        train_x = [self._extract_feats(t) for t in train_x]
        # from now on, all newly extracted features will be tertilized
        self._compute_tertiles(train_x)
        # but we need to take care of the previously extracted ones
        for feats_i in train_x:
            self._tertilize_feats(feats_i)
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
            if a_grid_search:
                self.classes_ = self._model.best_estimator_.classes_
            else:
                self.classes_ = self._model.classes_

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
        self._logger.debug("%s trained", self.name)

    @abc.abstractmethod
    def _extract_feats(self, a_tweet):
        """Method for training the model.

        Args:
          a_tweet (cgsa.utils.data.Tweet):
            training instance to extract features from

        Returns:
          dict: extracted features and their values

        """
        raise NotImplementedError

    def _compute_tertiles(self, feats, n=10):
        """Compute tertiles of feature values.

        Args:
          feats (list[dict]): list of extracted features
          n (int): number of tertiles to split the feature ranges into

        Returns:
          void

        """
        feats2vals = defaultdict(list)
        for feats_i in feats:
            for k, v in iteritems(feats_i):
                if isinstance(k, string_types) and k.startswith(ALEX):
                    feats2vals[k].append(v)
        marks = np.linspace(0, 100, num=n, endpoint=False)
        self._feats2tertiles = {
            k: np.percentile(v, marks, overwrite_input=True)
            for k, v in iteritems(feats2vals)}
