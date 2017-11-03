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

from csv import QUOTE_NONE
import abc
import pandas as pd

from cgsa.constants import INTENSIFIERS, SPACE_RE, USCORE_RE
from cgsa.base import BaseAnalyzer, ENCODING, SCORE
from cgsa.utils.trie import Trie


##################################################################
# Variables and Constants
BOUNDARIES = ["aber", "und", "oder", "weil", "denn", "während",
              "nachdem", "bevor", "als", "wenn", "obwohl",
              "jedoch", "obgleich", "wenngleich", "immerhin",
              "ob", "falls", "sofern", "wann", "welche", "welcher",
              "welchem", "welchen", "welches", "trotz", "dadurch",
              "damit", "daher", "deswegen", "dann", "folglich",
              "dementsprechend", "demnach", "deshalb", "somit",
              "somit", "daher", "hierdurch", "wo", "wobei", "dabei",
              "wohingegen", "wogegen", "bis",
              "außer", "dass"]
NEGATIONS = ["nicht", "kein", "keine", "keiner", "keinem", "keines", "keins",
             "weder", "nichts", "nie", "niemals", "niemand",
             "entbehren", "vermissen", "ohne", "Abwesenheit", "Fehlen",
             "Mangel", "frei von"]


##################################################################
# Classes
class LexiconBaseAnalyzer(BaseAnalyzer):
    """Abstract class for lexicon-based sentiment analysis.

    Attributes:

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        """Class constructor.

        """
        self._intensifiers = self._read_intensifiers(INTENSIFIERS)
        self._boundaries = self._words2trie(BOUNDARIES)
        self._negations = self._words2trie(NEGATIONS)

    def _read_intensifiers(self, int_fname=INTENSIFIERS, encoding=ENCODING):
        """Read intensifiers from file.

        Args:
          int_fname (str): path to the file containing intensifiers
          encoding (str): input encoding

        Returns:
          Trie: intensifier trie

        """
        INTENSIFIER = "intensifier"
        INT_TYPES = {INTENSIFIER: str, SCORE: float}

        int_list = pd.read_table(int_fname, header=None,
                                 names=(INTENSIFIER, SCORE),
                                 dtype=INT_TYPES, encoding=encoding,
                                 error_bad_lines=False, warn_bad_lines=True,
                                 keep_default_na=False, na_values=[''],
                                 quoting=QUOTE_NONE)
        intensifiers = Trie()
        for i, row_i in int_list.iterrows():
            term = USCORE_RE.sub(' ', row_i.intensifier)
            terms = SPACE_RE.split(self._preprocess(term))
            intensifiers.add(terms,
                             [None] * len(terms),
                             row_i.score)
        return intensifiers

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

    def _preprocess(self, a_txt):
        """Overwrite parent's method lowercasing strings.

        Args:
          a_txt (str): text to be preprocessed

        Returns:
          str: preprocessed text

        """
        return super(LexiconBaseAnalyzer, self)._preprocess(a_txt).lower()
