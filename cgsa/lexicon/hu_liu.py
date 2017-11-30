#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

##################################################################
# Variables and Constants
from cgsa.lexicon.base import LexiconBaseAnalyzer


##################################################################
# Classes
class HuLiuAnalyzer(LexiconBaseAnalyzer):
    """Lexicon-based sentiment analysis method of Hu and Liu (2004).

    """

    def __init__(self, lexicons=[], **kwargs):
        """Attributes:


        Notes:

        """
        super(HuLiuAnalyzer, self).__init__(lexicons)
        self.name = "hu-liu"

    def _compute_so(self, tweet):
        forms = [self._preprocess(w_i.form)
                 for w_i in tweet]
        self._logger.debug("forms: %r;", forms)
        lemmas = [self._preprocess(w_i.lemma)
                  for w_i in tweet]
        tags = [w_i.tag for w_i in tweet]
        match_input = [(f, l, t)
                       for f, l, t
                       in zip(forms, lemmas, tags)]
        # determine polar term matches
        polterm_matches = self._split_polterm_matches(
            tags,
            self._join_scores(
                self._polar_terms.search(match_input)))
        self._logger.debug("matched polar terms: %r",
                           polterm_matches)
        # match blocking constructs
        boundaries = self._find_boundaries(match_input)
        self._logger.debug("boundaries: %r", boundaries)
        # match negations
        neg_matches = []
        # neg_matches = [(start, end)
        #                for _, start, end
        #                in self._negations.search(match_input)]
        self._logger.debug("matched negations: %r", neg_matches)
        # in contrast to the original approach, we do not consider the polarity
        # of the previous tweet in case of a tie
        so_cnt = 0
        for pos, term_matches in zip(["noun", "verb", "adj", "adv"],
                                     [polterm_matches.nouns,
                                      polterm_matches.verbs,
                                      polterm_matches.adjectives,
                                      polterm_matches.adverbs]):
            for score_i, start_i, end_i in term_matches:
                prev_pos = start_i - 1
                # determine tokens, which come into consideration as negation
                neg_start, neg_end = self._find_negation(
                    prev_pos, neg_matches, boundaries,
                    forms, lemmas, tags, pos
                )
                if neg_start >= 0:
                    score_i *= -1.
                so_cnt += score_i  # 1 if score_i > 0 else -1
        return so_cnt
