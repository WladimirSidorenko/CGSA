#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from cgsa.lexicon.base import LexiconBaseAnalyzer
from six.moves import range


##################################################################
# Classes
class MustoAnalyzer(LexiconBaseAnalyzer):
    """Lexicon-based sentiment analysis method of Musto et al.

    Attributes:

    """

    def __init__(self, lexicons=[], **kwargs):
        """Class constructor.

        Args:
          lexicons (list[str]): list of sentiment lexicons

        Notes:

          Given a Tweet T, we split it in several micro-phrases m 1 . . . m n
          according to the splitting cues occurring in the content. As
          splitting cues we used punctuations, adverbs and conjunctions.
          Whenever a splitting cue is found in the text, a new micro-phrase is
          built.

          Given such a representation, we define the sentiment S conveyed by a
          Tweet T as the sum of the polarity conveyed by each of the
          micro-phrases m i which compose it. In turn, the polarity of each
          micro-phrase depends on the sentimental score of each term in the
          micro-phrase, labeled as score(t j ), which is obtained from one of
          the above described lexical resources. In this preliminary
          formulation of the approach we did not take into account any valence
          shifters [7] except of the negation. When a negation is found in the
          text, the polarity of the whole micro-phrase is inverted. No
          heuristics have been adopted to deal with neither language
          intensifiers and downtoners, or to detect irony

          Normalized-emphasized approach

          # score of the tweet is sum of the scores of each micro-phrase

          $$s(T) = \sum_i^M pol(m_i) $$

          # score of the micro-phrase is a normalized sum of the scores of
          # polar terms multiplied by the coefficient of their main part of
          # speech

          As regards the emphasis-based approach, the boosting factor w is set
          to 1.5 after a rough tuning (the score of adjectives, adverbs and
          nouns is increased by 50%).

          When a negation is found in the text, the polarity of the whole
          micro-phrase is inverted.

        """
        super(MustoAnalyzer, self).__init__(lexicons)
        self.name = "musto"

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
        # determine micro-phrase boundaries
        boundaries = self._find_boundaries(match_input)
        self._logger.debug("boundaries: %r (%s);",
                           boundaries,
                           ", ".join([forms[i]
                                      for i, j in boundaries]))
        # determine polar term matches
        polterm_matches = self._split_polterm_matches(
            tags, self._join_scores(
                self._polar_terms.search(match_input)))
        self._logger.debug("polterm_matches: %r;", polterm_matches)
        if not polterm_matches:
            return 0.
        # initialize a container for storing scores of every micro-phrase
        phrase_scores = [0.] * (len(boundaries) + 1)
        self._logger.debug("phrase_scores: %r;", phrase_scores)
        # length of each micro-phrase
        phrase_lengths = [0.] * len(phrase_scores)
        self._logger.debug("phrase_scores: %r;", phrase_scores)
        # initialize a mapping from word indinces to the index of micro-phrase
        w_idx2phr_idx = {}
        phr_idx = -1
        prev_start = 0
        for phr_idx, (bndr_start, bndr_end) in enumerate(boundaries):
            for w_idx in range(prev_start, bndr_start):
                w_idx2phr_idx[w_idx] = phr_idx
            phrase_lengths[phr_idx] = bndr_start - prev_start
            prev_start = bndr_end + 1
        # the very last micro-phrase might appear after the last boundary
        phr_idx += 1
        for w_idx in range(prev_start, len(forms)):
            w_idx2phr_idx[w_idx] = phr_idx
        phrase_lengths[phr_idx] = len(forms) - prev_start
        self._logger.debug("w_idx2phr_idx: %r; phrase_lengths: %r;",
                           w_idx2phr_idx, phrase_lengths)
        # add the score of each found polar temr to the respective micro-phrase
        for coeff, pol_terms in zip([1.5, 1.5, 1.5, 1.],
                                    [polterm_matches.adjectives,
                                     polterm_matches.adverbs,
                                     polterm_matches.nouns,
                                     polterm_matches.verbs]):
            for score_i, start_i, end_i in pol_terms:
                for w_idx in range(start_i, end_i + 1):
                    if w_idx not in w_idx2phr_idx:
                        continue
                    phr_idx = w_idx2phr_idx[w_idx]
                    phrase_scores[phr_idx] += score_i * coeff
        # inverse scores of micro-phrases if they have a negation
        negated_phrases = set()  # every phrase can be negated only once
        # for _, start, end in self._negations.search(match_input):
        #     for w_idx in range(start, end + 1):
        #         if w_idx in w_idx2phr_idx:
        #             phr_idx = w_idx2phr_idx[w_idx]
        #             if phr_idx in negated_phrases:
        #                 continue
        #             phrase_scores[phr_idx] *= -1.
        #             negated_phrases.add(phr_idx)
        #             break
        # normalize scores by the lengths of the micro-phrases
        self._logger.debug("phrase_scores: %r; phrase_lengths: %r;",
                           phrase_scores, phrase_lengths)
        total_score = sum(score_i / (length_i or 1e10)
                          for score_i, length_i
                          in zip(phrase_scores, phrase_lengths))
        self._logger.debug("total_score: %r;", total_score)
        return total_score
