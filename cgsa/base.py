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

from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
try:
    from cPickle import load
except ImportError:
    from _pickle import load
import abc
import numpy as np
import os
import pandas as pd
import re

from cgsa.common import LOGGER
from cgsa.constants import ENCODING, NFOLDS

##################################################################
# Variables and Constants
TERM = "term"
POLCLASS = "polclass"
SCORE = "score"
LEX_CLMS = (TERM, POLCLASS, SCORE)
LEX_TYPES = {TERM: str, POLCLASS: str, SCORE: float}
NEG_SFX = r"_NEG"
NEG_SFX_RE = re.compile(re.escape(re.escape(NEG_SFX)
                                  + r"(:?FIRST)?'"))
USCORE_RE = re.compile(r'_')
SPACE_RE = re.compile(r"\s\s+")
AT_RE = re.compile(r"\b@\w+", re.U)
URI_RE = re.compile(
    r"\b((?:[\w]{3,5}://?|(?:www|bit)[.]|(?:\w[-\w]+[.])+"
    r"(?:a(?:ero|sia|[c-gil-oq-uwxz])|b(?:iz|[abd-jmnorstvwyz])"
    r"|c(?:at|o(?:m|op)|[acdf-ik-orsuvxyz])|d[dejkmoz]|e"
    r"(?:du|[ceghtu])|f[ijkmor]|g(?:ov|[abd-ilmnp-uwy])"
    r"|h[kmnrtu]|i(?:n(?:fo|t)|[del-oq-t])|j(?:obs|[emop])"
    r"|k[eghimnprwyz]|l[abcikr-vy]|m(?:il|obi|useum"
    r"|[acdeghk-z])|n(?:ame|et|[acefgilopruz])|"
    r"o(?:m|rg)|p(?:ro|[ae-hk-nrstwy])|qa|r[eosuw]"
    r"|s[a-eg-or-vxyz]|t(?:(?:rav)?el|[cdfghj-pr])"
    r"|xxx)\b)(?:[^\s,.:;]|\.\w)*)")
CI_PRFX = "%%case-insensitive-"


##################################################################
# Classes
class BaseAnalyzer(object):
    """Abstract class for coarse-grained sentiment analyzers.

    Attributes:

    """
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def load(a_path):
        """Load model from disc and restore its members.

        Args:
          a_path (str): path to the serialized model

        """
        analyzer = load(a_path)
        analyzer._logger = LOGGER
        return analyzer

    def __init__(self, *args, **kwargs):
        """Class constructor.

        """
        self.name = "BaseAnalyzer"
        self._logger = LOGGER
        self._term2mnl = defaultdict(dict)
        self._neg_term2mnl = defaultdict(dict)
        self._term2auto = defaultdict(dict)
        self._neg_term2auto = defaultdict(dict)
        self._read_lexicons(self._term2mnl, self._neg_term2mnl,
                            kwargs.get("manual_lexicons"))
        self._read_lexicons(self._term2auto, self._neg_term2auto,
                            kwargs.get("auto_lexicons"))

    @abc.abstractmethod
    def train(self, train_x, train_y, dev_x, dev_y, grid_search=True):
        """Method for training the model.

        Args:
          train_x (list):
            list of training instances
          train_y (list):
            labels of the training instances
          dev_x (list):
            list of devset instances
          dev_y (list):
            labels of the devset instances
          grid_search (bool):
            use grid search in order to determine hyper-paramaters of
            the model

        Returns:
          void:

        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_proba(self, msg, yvec):
        """Method for predicting sentiment propbablities of a single message.

        Args:
          msg (cgsa.data.Tweet):
            discourse relation whose sense should be predicted
          yvec (np.array): target array for storing the probabilities

        Returns:
          void

        Note:
          modifies `'

        """
        raise NotImplementedError

    def reset(self):
        """Remove members which cannot be serialized.

        """
        self._logger = None

    @abc.abstractmethod
    def _extract_feats(self, a_tweet):
        """Method for training the model.

        Args:
          a_tweet (cgsa.data.Tweet):
            training instance to extract features from

        Returns:
          void:

        """
        raise NotImplementedError

    def _get_cv(self, train_x, train_y, dev_x, dev_y, n_folds=NFOLDS):
        """Generate cross-validator from training and development data.

        Args:
          train_x (list):
            list of training instances
          train_y (list):
            labels of the training instances
          dev_x (list):
            list of devset instances
          dev_y (list):
            labels of the devset instances
          n_folds (int):
            number of folds

        Returns:
          3-tuple: cross validator, training instances, training labels

        """
        if dev_y:
            return self._get_devset_cv(self, train_x, train_y,
                                       dev_x, dev_y, n_folds)
        cv = None
        return cv, train_x, train_y

    def _get_devset_cv(self, train_x, train_y, dev_x, dev_y, n_folds):
        """Generate a cross-validator from training and development data.

        Args:
          train_x (list):
            list of training instances
          train_y (list):
            labels of the training instances
          dev_x (list):
            list of devset instances
          dev_y (list):
            labels of the devset instances
          n_folds (int):
            number of folds

        Returns:
          3-tuple: cross validator, training instances, training labels

        """
        folds = []
        n_train = len(train_y)
        n_dev = len(dev_y)
        dev_ids = [n_train + i for i in xrange(n_dev)]
        # create stratified K-folds over the training data
        skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True)
        for train_ids, test_ids in skf.split(train_x, train_y):
            folds.append((train_ids,
                          np.concatenate((test_ids, dev_ids))))
        train_x += dev_x
        train_y += dev_y
        return folds, train_x, train_y

    def _read_lexicons(self, a_pos_term2polscore, a_neg_term2polscore,
                       a_lexicons, a_encoding=ENCODING, a_cs_fallback=False):
        """Load lexicons.

        Args:
          a_pos_term2polscore (dict): mapping from terms to their polarity
            scores
          a_neg_term2polscore (dict): mapping from negated terms to their
            polarity scores
          a_lexicons (list): tags of the input instance
          a_encoding (str): input encoding
          a_cs_fallback (bool): use case-sensitive fallback

        Returns:
          void:

        Note:
          populates `a_pos_term2polscore` and `a_neg_term2polscore` in place

        """
        for lexpath_i in a_lexicons:
            lexname = os.path.splitext(os.path.basename(
                lexpath_i
            ))[0]
            LOGGER.debug(
                "Reading lexicon %s...", lexname
            )
            lexicon = pd.read_table(lexpath_i, header=None, names=LEX_CLMS,
                                    dtype=LEX_TYPES, encoding=a_encoding,
                                    error_bad_lines=False, warn_bad_lines=True)
            for i, row_i in lexicon.iterrows():
                term = USCORE_RE.sub(' ', row_i.term)
                if NEG_SFX_RE.search(term):
                    term = NEG_SFX_RE.sub("", term)
                    trg_lex = a_neg_term2polscore
                else:
                    trg_lex = a_pos_term2polscore
                term = self._preprocess(term)
                lex_key = (lexname, row_i.polclass)
                if a_cs_fallback:
                    term_key = CI_PRFX + term.lower()
                    if lex_key not in trg_lex[term_key]:
                        trg_lex[term_key][lex_key] = []
                    trg_lex[term_key][lex_key].append(row_i.score)
                else:
                    term = term.lower()
                    if lex_key not in trg_lex[term]:
                        trg_lex[term][lex_key] = []
                    trg_lex[term][lex_key].append(row_i.score)
            LOGGER.debug(
                "Lexicon %s read...", lexname
            )

    def _preprocess(self, a_txt):
        """Replace twitter-specific phenomena.

        Args:
          a_txt (str): text to be preprocessed

        Returns:
          str: preprocessed text

        """
        a_txt = SPACE_RE.sub(' ', a_txt).strip()
        a_txt = AT_RE.sub("@someuser", a_txt)
        a_txt = URI_RE.sub("http://someuri", a_txt)
        return a_txt
