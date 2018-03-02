#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from bisect import bisect_left
from collections import defaultdict
from six import iteritems, string_types
from six.moves import xrange
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import abc
import logging
import numpy as np
import pandas as pd
import re

from cgsa.base import BaseAnalyzer
from cgsa.constants import IDX2CLS
from cgsa.utils.common import LOGGER

##################################################################
# Variables and Constants
ALEX = "alex_"
DFLT_C = 0.01
DFLT_CLS_WGHT = None
DFLT_PARAMS = {"class_weight": DFLT_CLS_WGHT, "loss": "hinge",
               "penalty": "l1", "dual": True,
               "multi_class": "crammer_singer"}
PARAM_GRID = {"clf__C": np.linspace(1e-2, 1, 7)}
WLDCARD = '*'
NEG_SFX_RE = re.compile(r"_NEG", re.I)


##################################################################
# Class
class MLBaseAnalyzer(BaseAnalyzer):
    """Class for ML-based sentiment analysis.

    Attributes:

    """

    def __init__(self, lexicons, a_clf=None, **kwargs):
        """Class constructor.

        Args:
          kwargs (dict): additional keyword arguments

        """
        super(MLBaseAnalyzer, self).__init__(**kwargs)
        self.name = "MLBaseAnalyzer"
        # read lexicons
        self.N_JOBS = 1
        self._term2mnl = defaultdict(dict)
        self._neg_term2mnl = defaultdict(dict)
        self._term2auto = defaultdict(dict)
        self._neg_term2auto = defaultdict(dict)
        self._read_lexicons({"manual": (self._term2mnl, self._neg_term2mnl),
                             "auto": (self._term2auto, self._neg_term2auto)},
                            lexicons)
        # set up classifier
        clf = a_clf or LinearSVC(C=DFLT_C, **DFLT_PARAMS)
        self._model = Pipeline([("vect", DictVectorizer()),
                                ("clf", clf)])
        self.PARAM_GRID = PARAM_GRID
        self._cs_fallback = False
        self._feats2tertiles = {}
        self._feats2weights = None

    def predict_proba(self, msg, yvec):
        feats = self._extract_feats(msg)
        if self._logger.getEffectiveLevel() <= logging.DEBUG:
            self._logger.debug(
                "Top-10 Most Relevant Features:\n%s",
                self._get_top_feats(
                    feats
                )
            )
            pass
        dec = self._model.decision_function(feats)
        if len(dec.shape) > 1:
            dec = np.mean(dec, axis=0)
        for i, ival in enumerate(dec):
            yvec[self.classes_[i]] += ival

    def reset(self):
        """Remove members which cannot be serialized.

        """
        super(MLBaseAnalyzer, self).reset()
        self._feats2weights = None

    def restore(self):
        """Restore members which could not be serialized.

        """
        self._logger = LOGGER
        self._load_feats_weights()

    def train(self, train_x, train_y, dev_x, dev_y, a_grid_search,
              a_extract_feats=True):
        self._logger.debug("Training %s...", self.name)
        train_len = len(train_x)
        if a_extract_feats:
            train_x = [self._extract_feats(t) for t in train_x]
        # from now on, all newly extracted features will be tertilized
        self._compute_tertiles(train_x)
        # but we need to take care of the previously extracted ones
        for feats_i in train_x:
            self._tertilize_feats(feats_i)
        if a_extract_feats:
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

    def _extract_ngrams(self, a_feats, a_input, a_min_len, a_max_len,
                        a_skip_grams=False, tags=None, a_wght=1.):
        """Extract n-grams up to length n from iterable.

        Args:
          a_feats (dict): target dictionary of features to populate
          a_input (iterable): input sequence to extract the n-grams from
          a_min_len (int): minimum length of an n-gram
          a_max_len (int): maximum length of an n-gram
          a_skip_grams (bool): create skip grams (replace one of the
            tokens with wildcard)
          a_tags (list or None): optional list of part-of-speech tags
          a_wght (float): default feature weight

        Returns:
          set: extracted n-grams

        """
        ret = set()
        last_toks = []
        mmax = a_max_len + 1
        for i, x in enumerate(a_input):
            last_toks.append((i, x))
            if len(last_toks) > a_max_len:
                last_toks.pop(0)
            for i in xrange(a_min_len, mmax):
                ngram = last_toks[-i:]
                ret.add(tuple(ngram))
                if a_skip_grams and i > 1:
                    for j in xrange(len(ngram)):
                        tmp = ngram[j]
                        ngram[j] = (tmp[0], WLDCARD)
                        ret.add(tuple(ngram))
                        ngram[j] = tmp
        for ng in ret:
            a_feats['-'.join(n[-1] for n in ng)] = a_wght
        if tags:
            new_ret = set((tuple(n[0] for n in ng),
                           tuple(n[1] for n in ng),
                           tuple(tags[n[0]] for n in ng))
                          for ng in ret)
        else:
            new_ret = set((tuple(n[0] for n in ng),
                           tuple(n[1] for n in ng),
                           tuple())
                          for ng in ret)
        return new_ret

    def _get_tertile(self, val, tertiles):
        """Determine the tertile of the given value.

        Args:
          val (float): feature value to determine the tertile for
          tertiles (np.array): array of tertile marks

        Returns:
          int: tertile of the given feature

        """
        return max(0, bisect_left(tertiles, val) - 1)

    def _tertilize_feats(self, feats):
        """Split specific features into separate attributes by their tertiles.

        Args:
          feats (dict): original features to be split

        Returns:
          void:

        Note:
          modifies `feats` in-place

        """
        for feat_name, tertiles in iteritems(self._feats2tertiles):
            if feat_name in feats:
                val = feats.pop(feat_name)
                tertile = self._get_tertile(val, tertiles)
                feats[feat_name + '-' + str(tertile)] = 1  # val

    def _get_top_feats(self, feats, n=10):
        """Obtain coefficients for extracted features .

        Args:
          feats (dict): features of an input istance
          n (int): top N highest-ranked features to display

        Returns:
          str: summmary of top-N most relevant features

        """
        if not self._feats2weights:
            self._load_feats_weights()
        ret = []
        for feat_name, feat_value in iteritems(feats):
            if feat_name in self._feats2weights:
                for cls, wght in iteritems(self._feats2weights[feat_name]):
                    ret.append((feat_name, cls, feat_value * wght))
        ret.sort(key=lambda x: abs(x[-1]), reverse=True)
        return '\n'.join(
            str(i) + ") " + feat_name + " (" + cls + "): " + str(feat_wght)
            for i, (feat_name, cls, feat_wght) in enumerate(ret[:n], 1)
        )

    def _load_feats_weights(self):
        """obtain feature weights from model.

        Args:

        Returns:
          void:

        """
        self._feats2weights = defaultdict(lambda: defaultdict(tuple))
        if hasattr(self._model, "best_estimator_"):
            steps = dict(self._model.best_estimator_.steps)
            vectorizer = steps["vect"]
            idx2feat_name = {idx: feat_name
                             for feat_name, idx
                             in iteritems(vectorizer.vocabulary_)}
            clf = steps["clf"]
            for i, feat_weights in enumerate(clf.coef_.T):
                feat_name = idx2feat_name[i]
                for j, wght_ij in enumerate(np.nditer(feat_weights)):
                    label = IDX2CLS[j]
                    self._feats2weights[feat_name][label] = wght_ij
