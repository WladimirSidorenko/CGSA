#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from itertools import chain
import pandas as pd
import os

from cgsa.base import (LEX_CLMS, LEX_TYPES, NEG_SFX_RE,
                       USCORE_RE, QUOTE_NONE)
from cgsa.constants import ENCODING, SPACE_RE
from cgsa.lexicon.base import LexiconBaseAnalyzer
from cgsa.utils.common import LOGGER
from cgsa.utils.trie import Trie

##################################################################
# Variables and Constants


##################################################################
# Class
class TaboadaAnalyzer(LexiconBaseAnalyzer):
    """Class for lexicon-based sentiment analysis.

    Attributes:

    """

    def __init__(self, lexicons=[]):
        """Class constructor.

        Args:
          lexicons (list[str]): list of sentiment lexicons

        """
        assert lexicons, \
            "Provide at least one lexicon for lexicon-based method."
        self._logger = LOGGER
        self._term2score = Trie(a_ignorecase=True)
        self._negterm2score = Trie(a_ignorecase=True)
        self._read_lexicons(self._term2score, self._negterm2score,
                            lexicons, a_encoding=ENCODING)

    def train(self, train_x, train_y, dev_x, dev_y,
              **kwargs):
        # no training is required for this method
        scores = [self._compute_so(tweet_i)
                  for tweet_i in chain(train_x, dev_x)]
        labels = [label_i
                  for label_i in chain(train_y, dev_y)]
        self._optimize_thresholds(scores, labels)

    def predict(self, msg):
        # no training is required for this method
        raise NotImplementedError

    def _compute_so(self, tweet):
        """Compute semantic orientation of a tweet.

        Args:
          tweet (cgsa.utils.data.Tweet): input message

        Returns:
          float: semantic orientation score

        """
        total_so = 0.
        total_cnt = 0
        toks = [self._preprocess(w_i.form) for w_i in tweet]
        lemmas = [self._preprocess(w_i.lemma) for w_i in tweet]
        tags = [w_i.tag for w_i in tweet]
        assert len(toks) == len(lemmas), \
            "Unmatching number of tokens and lemmas."
        assert len(toks) == len(tags), "Unmatching number of tokens and tags."
        term_matches = self._term2score.match(
            zip(toks, lemmas, tags))
        self._logger.debug("term_matches: %r", term_matches)
        negterm_matches = self._negterm2score.match(
            zip(toks, lemmas, tags))
        self._logger.debug("negterm_matches: %r", negterm_matches)
        import sys
        sys.exit(66)
        total_so = total_so / (float(total_cnt) or 1e10)

    def _optimize_thresholds(self, scores, labels):
        """Compute optimal thershold values

        Args:
          scores (list[float]): SO scores assigned to instances
          labels (list[str]): gold labels

        Returns:
          void: optimizes instance attributes in place

        """
        raise NotImplementedError

    def _read_lexicons(self, a_pos_term2polscore, a_neg_term2polscore,
                       a_lexicons, a_encoding=ENCODING):
        """Load lexicons.

        Args:
          a_pos_term2polscore (cgsa.utils.trie.Trie): mapping from terms to
            their polarity scores
          a_neg_term2polscore (cgsa.utils.trie.Trie): mapping from negated
            terms to their polarity scores
          a_lexicons (list): tags of the input instance
          a_encoding (str): input encoding

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
                                    error_bad_lines=False, warn_bad_lines=True,
                                    keep_default_na=False, na_values=[''],
                                    quoting=QUOTE_NONE)
            for i, row_i in lexicon.iterrows():
                term = USCORE_RE.sub(' ', row_i.term)
                if NEG_SFX_RE.search(term):
                    term = NEG_SFX_RE.sub("", term)
                    trg_lex = a_neg_term2polscore
                else:
                    trg_lex = a_pos_term2polscore
                term = self._preprocess(term)
                trg_lex.add(SPACE_RE.split(term), SPACE_RE.split(row_i.pos),
                            (lexname, row_i.polarity, row_i.score))
            LOGGER.debug(
                "Lexicon %s read...", lexname
            )

    def _preprocess(self, a_txt):
        """Overwrite parent's method lowercasing strings.

        Args:
          a_txt (str): text to be preprocessed

        Returns:
          str: preprocessed text

        """
        return super(TaboadaAnalyzer, self)._preprocess(a_txt).lower()
