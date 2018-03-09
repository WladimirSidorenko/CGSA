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

from bisect import bisect_left
from collections import defaultdict
from csv import QUOTE_NONE
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

from cgsa.utils.common import LOGGER
from cgsa.utils.trie import Trie
from cgsa.constants import (BOUNDARIES, ENCODING, NEGATIONS,
                            NFOLDS, PUNCT_RE, SPACE_RE, SSPACE_RE,
                            USCORE_RE)

##################################################################
# Variables and Constants
TERM = "term"
POS = "pos"                # part of speech tags
POLARITY = "polarity"
SCORE = "score"
SENT_PUNCT_RE = re.compile(r"^[.;:!?]$")
LEX_CLMS = (TERM, POS, POLARITY, SCORE)
LEX_TYPES = {TERM: str, POS: str, POLARITY: str, SCORE: float}
NEG_SFX = r"_NEG"
NEG_SFX_RE = re.compile(re.escape(re.escape(NEG_SFX)
                                  + r"(:?FIRST)?'"))
SZET_RE = re.compile('ß')
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
        analyzer._load()
        return analyzer

    def __init__(self, *args, **kwargs):
        """Class constructor.

        """
        self.name = "BaseAnalyzer"
        self._logger = LOGGER
        self._term2lex = defaultdict(dict)
        self._neg_term2lex = defaultdict(dict)
        self._boundaries = self._words2trie(BOUNDARIES)
        self._negations = self._words2trie(NEGATIONS)
        self._read_lexicons({"any": (self._term2lex, self._neg_term2lex)},
                            kwargs.get("lexicons", []))

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
          msg (cgsa.utils.data.Tweet):
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
        self._boundaries.reset()
        self._negations.reset()

    def _load(self):
        """Re-initialize reset members.

        """
        self._logger = LOGGER

    @abc.abstractmethod
    def _extract_feats(self, a_tweet):
        """Method for training the model.

        Args:
          a_tweet (cgsa.utils.data.Tweet):
            training instance to extract features from

        Returns:
          void:

        """
        raise NotImplementedError

    def _find_boundaries(self, match_input):
        """Determine boundaries which block propagation.

        Args:
          match_input (list[tuple]): list of tuples comprising forms, lemmas,
            and part-of-speech tags

        Returns:
          list[tuple]: indices of matched boundaries

        """
        boundaries = self._boundaries.search(match_input)
        for i, (tok_i, _, _) in enumerate(match_input):
            if PUNCT_RE.search(tok_i):
                boundaries.append((None, i, i))
        boundaries = [(start, end)
                      for _, start, end
                      in self._boundaries.select_llongest(boundaries)]
        return boundaries

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
            return self._get_devset_cv(train_x, train_y,
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

    def _get_sent_punct(self, index, forms, boundaries):
        """Find closest punctuation mark on the right from index.

        Args:
          index (int): index of the polar term
          forms (list[str]): original tweet tokens
          boundaries (list[int]): boundary tokens

        Returns:
          str: closest punctuation mark on the right or empty string

        """
        idx = bisect_left(boundaries, (index, index))
        for boundary in boundaries[idx:]:
            tok = forms[boundary[0]]
            if SENT_PUNCT_RE.match(tok):
                return tok
        return ""

    def _read_lexicons(self, a_lextype2lex, a_lexicons, a_encoding=ENCODING):
        """Load lexicons.

        Args:
          a_lextype2lex (dict: lextype -> (dict, dict)): mapping from
            lexicon type to target dictionaries for storing terms
          a_lexicons (list): tags of the input instance
          a_encoding (str): input encoding

        Returns:
          void:

        Note:
          populates `a_pos_term2polscore` and `a_neg_term2polscore` in place

        """
        for lexpath_i in a_lexicons:
            fname = os.path.basename(lexpath_i).split('.')
            lexname = fname[0]
            lextype = fname[-2] if len(fname) > 1 else ""
            self._logger.debug("Reading lexicon %s...", lexname)
            if lextype not in a_lextype2lex:
                if "any" in a_lextype2lex:
                    pos_term2polscore, neg_term2polscore = a_lextype2lex["any"]
                else:
                    self._logger.error("Unknown lexicon type: %s." % lextype)
                    raise NotImplementedError
            else:
                # determine target dictionaries for storing positive and
                # negative terms
                pos_term2polscore, neg_term2polscore = a_lextype2lex[lextype]
            lexicon = pd.read_table(lexpath_i, header=None, names=LEX_CLMS,
                                    dtype=LEX_TYPES, encoding=a_encoding,
                                    error_bad_lines=False, warn_bad_lines=True,
                                    keep_default_na=False, na_values=[''],
                                    quoting=QUOTE_NONE)
            for i, row_i in lexicon.iterrows():
                term = USCORE_RE.sub(' ', row_i.term)
                if NEG_SFX_RE.search(term):
                    term = NEG_SFX_RE.sub("", term)
                    trg_lex = neg_term2polscore
                else:
                    trg_lex = pos_term2polscore
                term = self._preprocess(term)
                lex_key = (lexname, row_i.polarity)

                term = term.lower()
                if lex_key not in trg_lex[term]:
                    trg_lex[term][lex_key] = [row_i.score]
                else:
                    trg_lex[term][lex_key].append(row_i.score)
            self._logger.debug(
                "Lexicon %s read...", lexname
            )

    def _preprocess(self, a_txt):
        """Replace twitter-specific phenomena.

        Args:
          a_txt (str): text to be preprocessed

        Returns:
          str: preprocessed text

        """
        a_txt = SZET_RE.sub("ss", a_txt).strip()
        a_txt = SSPACE_RE.sub(' ', a_txt).strip()
        a_txt = AT_RE.sub("@someuser", a_txt)
        a_txt = URI_RE.sub("http://someuri", a_txt)
        return a_txt

    def _words2trie(self, words):
        """Convert collection of words to a trie.

        Args:
          words (iterable): collection of untagged words

        Returns:
          (cgsa.utils.trie.Trie): trie

        """
        ret = Trie()
        for w_i in words:
            term = USCORE_RE.sub(' ', w_i)
            terms = SPACE_RE.split(self._preprocess(term))
            ret.add(terms, [None] * len(terms), 1.)
        return ret
