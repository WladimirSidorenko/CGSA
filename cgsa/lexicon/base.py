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

from cgsa.constants import PUNCT_RE
from cgsa.base import BaseAnalyzer
import abc


##################################################################
# Variables and Constants
BOUNDARY_WORDS = set(["aber", "und", "oder", "weil", "denn", "während",
                      "nachdem", "bevor", "als", "wenn", "obwohl",
                      "jedoch", "obgleich", "wenngleich", "immerhin",
                      "ob", "falls", "sofern", "wann", "welche", "welcher",
                      "welchem", "welchen", "welches", "trotz", "dadurch",
                      "damit", "daher", "deswegen", "dann", "folglich",
                      "dementsprechend", "demnach", "deshalb", "somit",
                      "somit", "daher", "hierdurch", "wo", "wobei", "dabei",
                      "wohingegen", "wogegen", "bis",
                      "außer", "dass"])


##################################################################
# Classes
class LexiconBaseAnalyzer(BaseAnalyzer):
    """Abstract class for lexicon-based sentiment analysis.

    Attributes:

    """
    __metaclass__ = abc.ABCMeta

    def at_boundary(snt, idx):
        ret = (idx == len(snt) - 1
               or PUNCT_RE.match(snt[idx])
               or snt[idx] in BOUNDARY_WORDS)
        return ret

    def find_negation(index, word_type):
        """Look for negations appearing in tweet.


        """
        search = True
        found = -1
        while search and not at_boundary(index) and index >= 0:
            current = get_word(text[index]).lower()
            if current in negators:
                search = False
                found = index
                if restricted_neg[word_type] \
                   and current not in skipped[word_type] \
                   and get_tag(text[index]) not in skipped[word_type]:
                    search = False
            index -= 1
        return found
