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
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from bisect import bisect_left
from itertools import chain
from sklearn.metrics import f1_score
from six import iterkeys
import abc
import numpy as np
import pandas as pd
import os
import re

from cgsa.constants import (BOUNDARIES, CLS2IDX, IDX2CLS, INTENSIFIERS,
                            NEGATIONS, PUNCT_RE, SPACE_RE, USCORE_RE)
from cgsa.base import (BaseAnalyzer, ENCODING, LEX_CLMS, LEX_TYPES,
                       NEG_SFX_RE, QUOTE_NONE)
from cgsa.utils.common import LOGGER
from cgsa.utils.trie import Trie


##################################################################
# Variables and Constants
PRIMARY_LABEL_SCORE = 0.51
SECONDARY_LABEL_SCORE = (1. - PRIMARY_LABEL_SCORE) / float(len(CLS2IDX) - 1)
SKIP = {"adj": set(["selbst", "sogar", "zu", "sein", "bin", "bist", "ist",
                    "sind", "seid", "war", "warst", "wart", "waren", "wäre",
                    "wärest", "wäret", "wären", "habe", "hast", "hat",
                    "haben", "habt", "gehabt", "hätte", "hättest", "hätten",
                    "hättet", "hätte", "hättest", "hätten", "hättet",
                    "mache", "machst", "macht", "machen", "machte", "machtest",
                    "machst", "machst", "machst", "machst", "machst", "done",
                    "des", "der", "als", "ART", "PPOSAT"]),
        "adv": set(["VVFIN", "VVIMP", "VVINF", "VVIZU", "VVPP", "VAFIN",
                    "VAIMP", "VAINF", "VAPP", "VMFIN", "VMINF", "VMPP"]),
        "verb": set(["PTKZU", "sein", "bin", "bist", "ist",
                     "sind", "seid", "war", "warst", "wart", "waren", "wäre",
                     "wärest", "wäret", "wären", "habe", "hast", "hat",
                     "haben", "habt", "hätte", "hättest", "hätten", "hättet",
                     "hätte", "hättest", "hätten", "hättet"]),
        "noun": set(["PWAT", "ART", "PDAT", "PIAT", "PIDAT", "PDAT", "PPOSAT",
                     "PRELAT", "PWAT", "TRUNC", "NN", "NE", "der", "des",
                     "von", "habe", "hast", "hat", "haben", "habt", "gehabt",
                     "hätte", "hättest", "hätten", "hättet", "hätte",
                     "hättest", "hätten", "hättet", "komme", "kommst", "kommt",
                     "kommen", "komme", "kommst", "komme", "kommst", "mit",
                     "enthalten", "enthalte", "enthältst", "enthält",
                     "enthaltet", "enthaltete", "enthaltetest", "enthaltetet",
                     "enthalteten", "enthaltetet", "enthalten", "enthalten"])}


##################################################################
# Classes
class LexiconBaseAnalyzer(BaseAnalyzer):
    """Abstract class for lexicon-based sentiment analysis.

    """
    __metaclass__ = abc.ABCMeta

    class PolTermMatches(object):
        """Class comprising matches pertaining to specific parts of speech.

        """
        def __init__(self):
            """Class constructor."""
            self.adjectives = []
            self.adverbs = []
            self.nouns = []
            self.verbs = []
            # odered mapping from part-of-speech tags to corresponding
            # containers for matches
            self.TAGS2CONTAINER = [
                (set(["VAFIN", "VAIMP", "VAIMP", "VAINF", "VAPP",
                      "VMFIN", "VMINF", "VVFIN", "VVIMP", "VVINF",
                      "VVIZU", "VVPP"]), self.verbs),
                (set(["NE", "NN", "FM", "XY"]), self.nouns),
                (set(["ADJA", "ADJD"]), self.adjectives)
                # adverbs will be used by default for all remaining cases
            ]

        def __repr__(self):
            ret = ("<{:s}: adjectives: {!r}; adverbs: {!r};"
                   " nouns: {!r}; verbs: {!r}>").format(
                       self.__class__.__name__,
                       self.adjectives, self.adverbs,
                       self.nouns, self.verbs)
            return ret

    def __init__(self, lexicons=[], **kwargs):
        """Class constructor.

        Args:
          lexicons (list[str]): list of sentiment lexicons
          kwargs (dict): keyword arguments

        """
        assert lexicons, \
            "Provide at least one lexicon for lexicon-based method."
        self._logger = LOGGER
        self._intensifiers = self._read_intensifiers(INTENSIFIERS)
        self._boundaries = self._words2trie(BOUNDARIES)
        self._negations = self._words2trie(NEGATIONS)
        self._thresholds = None
        self._polar_terms = self._read_lexicons(lexicons,
                                                a_encoding=ENCODING)

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
        score = self._compute_so(msg)
        self._logger.debug("score: %f, thresholds: %r;",
                           score, self._thresholds)
        label = bisect_left(self._thresholds, score)
        self._logger.debug("score: %f, label: %d, yvec: %r;",
                           score, label, yvec)
        yvec[:] = SECONDARY_LABEL_SCORE
        yvec[label] = PRIMARY_LABEL_SCORE
        self._logger.debug("resulting yvec: %r;", yvec)

    def reset(self):
        """Remove members which cannot be serialized.

        """
        super(LexiconBaseAnalyzer, self).reset()
        self._logger = None
        self._intensifiers.reset()
        self._polar_terms.reset()

    def restore(self):
        """Restore members which could not be serialized.

        """
        self._logger = LOGGER
        self._intensifiers.restore()
        self._boundaries.restore()
        self._negations.restore()
        self._polar_terms.restore()

    def train(self, train_x, train_y, dev_x, dev_y,
              **kwargs):
        # no training is required for this method
        scores = [self._compute_so(tweet_i)
                  for tweet_i in chain(train_x, dev_x)]
        labels = [label_i
                  for label_i in chain(train_y, dev_y)]
        # check thresholds for every score type
        f1, self._thresholds = self._optimize_thresholds(
            scores, labels
        )
        self._logger.info("F1: %f; thresholds: %r;",
                          f1, self._thresholds)

    @abc.abstractmethod
    def _compute_so(self, tweet):
        """Compute semantic orientation of a tweet.

        Args:
          tweet (cgsa.utils.data.Tweet): input message

        Returns:
          tuple: total SO score, total SO count, average SO value

        """
        raise NotImplementedError

    def _find_negation(self, index,
                       neg_matches, boundaries,
                       forms, lemmas, tags, word_type):
        """Look for negations appearing in the nearby context.

        Args:
          index (int): index of the potentially negated word
          neg_matches (list[tuple]): negation matches
          boundaries (list[tuple]): boundary matches
          forms (list[str]): tokens of the analyzed message
          lemmas (list[str]): lemmas of the analyzed message
          tags (list[str]): tags of the analyzed message
          word_type (str): part of speech of the potentially negated
            word

        Returns:
          bool: true if negation was found

        """
        not_found = (-1, -1)
        search_idx = (index, index)
        # check if there is any negation preceding the polar term
        neg_idx = bisect_left(neg_matches, search_idx) - 1
        if neg_idx < 0:
            return not_found
        neg_start, neg_end = neg_matches[neg_idx]
        # check if there is a boundary between polar term and negation
        boundary = self._find_next_boundary(index, boundaries)
        if boundary > neg_end:
            self._logger.debug("negation prevented by blocking")
            return not_found
        # otherwise, check if every token between the negtion and the polar
        # term can be skipped
        skip_items = SKIP["noun"]
        for i in range(neg_end + 1, index):
            for item_i in (forms[i], lemmas[i], tags[i]):
                if item_i in skip_items:
                    continue
            self._logger.debug(
                "negation prevented by non-skipped term: %r, %r, %r",
                forms[i], lemmas[i], tags[i]
            )
            return not_found
        return (neg_start, neg_end)

    def _find_next_boundary(self, index, boundaries, left=True):
        """Find nearest boundary on the left

        Args:
          index (int): position to find the nearest boundary for
          boundaries (list[tuple]): positions of boundaries
          left (bool): find left-most boundary (otherwise, right-most boundary
            is searched)

        Return:
          int: position of the nearest boundary on the left from index

        """
        boundary_idx = bisect_left(boundaries, (index, index))
        if boundary_idx < len(boundaries):
            boundary_start, boundary_end = boundaries[boundary_idx]
            if boundary_start <= index <= boundary_end:
                return index
        if left:
            boundary_idx -= 1
        if 0 <= boundary_idx < len(boundaries):
            _, boundary_end = boundaries[boundary_idx]
            return boundary_end
        return -1

    def _join_scores(self, matches):
        """Sum scores of all matches.

        Args:
          matches (list[tuple]): list of scores, starting and ending positions
            of matches

        Returns:
          list[tuple]: the same matches, but with all scores for a single match
            summed

        """
        return [(sum(res[-1] for res in results), start, end)
                for results, start, end in matches]

    def _load_cond_probs(self, cond_prob_fname):
        """Load conditional probabilities of lexicon terms.

        Args:
          cond_prob_fname (str): path to the file containing conditional
            probabilities

        Returns:
          dict: positive and negative conditional probabilitis of lexicon terms

        """
        prob_table = pd.read_table(cond_prob_fname, header=None,
                                   names=("term", "positive", "negative"),
                                   dtype={"term": str,
                                          "positive": float,
                                          "negative": float},
                                   encoding=ENCODING,
                                   error_bad_lines=False,
                                   warn_bad_lines=True,
                                   keep_default_na=False,
                                   na_values=[''],
                                   quoting=QUOTE_NONE)
        cond_probs = {}
        for _, row_i in prob_table.iterrows():
            # Jurek et al. use scores in the range [-100, 100]
            probs = (row_i.positive * 100, row_i.negative * 100)
            cond_probs[row_i.term] = probs
            if row_i.term != row_i.term.lower():
                cond_probs[row_i.term.lower] = probs
        return cond_probs

    def _read_intensifiers(self, intensifiers=INTENSIFIERS):
        """Read intensifiers from file.

        Args:
          intensifiers (pandas.Dataframe): path to the file containing
            intensifiers

        Returns:
          Trie: intensifier trie

        """
        itrie = Trie()
        final_states = set()
        for i, row_i in intensifiers.iterrows():
            term = USCORE_RE.sub(' ',
                                 row_i.intensifier)
            terms = SPACE_RE.split(self._preprocess(term))
            final_states.add(
                itrie.add(terms,
                          [None] * len(terms),
                          row_i.score))
        for state_i in final_states:
            state_i.classes = sum(state_i.classes)
        return itrie

    def _read_lexicons(self, a_lexicons, a_encoding=ENCODING):
        """Load lexicons.

        Args:
          a_lexicons (list): tags of the input instance
          a_encoding (str): input encoding

        Returns:
          cgsa.utils.trie.Trie: constructed polar terms trie

        """
        ret = Trie(a_ignorecase=True)
        for lexpath_i in a_lexicons:
            lexname = os.path.splitext(os.path.basename(
                lexpath_i
            ))[0]
            self._logger.debug(
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
                    # Taboada's method explicitly accounts for negations, so we
                    # skip negated entries from the lexicon altogether
                    continue
                term = self._preprocess(term)
                ret.add(SPACE_RE.split(term),
                        SPACE_RE.split(row_i.pos),
                        (lexname, row_i.polarity, row_i.score))
            self._logger.debug(
                "Lexicon %s read...", lexname
            )
        return ret

    def _optimize_thresholds(self, scores, gold_labels):
        """Exhaustively search for the best threshold values.

        Args:
          scores (np.array): SO scores assigned to instances
          gold_labels (list[str]): gold labels to compare with

        Returns:
          float, tuple: maximum achieved F1 score, threshold values

        """
        self._logger.debug("oprimizing threshold values: "
                           "scores: %r; gold_labels: %r",
                           scores, gold_labels)
        assert len(scores) == len(gold_labels), \
            "Unmatching number of predicted and gold labels."
        thresholds = []
        best_f1 = -1.
        score_space = list(set(scores))
        score_space.sort()
        scores = np.array(scores)
        n_instances = len(scores)
        predicted_labels = np.zeros(n_instances)

        tagset = [k for k in iterkeys(IDX2CLS)]
        tagset.sort()
        assert len(tagset) - 1 <= len(set(score_space)), \
            "Not enough scores to determine threshold for all labels."
        prev_threshold_idx = 1
        prev_threshold = score_space[0] - 1
        # find best split point for each adjacent pair of labels
        for i in range(0, len(tagset) - 1):
            label1, label2 = tagset[i], tagset[i + 1]
            self._logger.debug(
                "Considering label pair: %r %r", label1, label2
            )
            best_threshold = -1.
            best_threshold_idx = -1
            for i, score_i in enumerate(score_space[prev_threshold_idx:],
                                        prev_threshold_idx):
                self._logger.debug(
                    "Considering threshold: %r", score_i
                )
                # temporarily set all predicted labels whose scores are between
                # `prev_threshold` and `score_i` to `label1`, and set all
                # labels whose scores are greater than `score_i` to `label2`
                predicted_labels[(prev_threshold < scores)
                                 & (scores <= score_i)] = label1
                predicted_labels[scores > score_i] = label2
                # self._logger.debug(
                #     "Predicted labels: %r", predicted_labels
                # )
                # estimate new F1 score
                f1 = f1_score(
                    predicted_labels, gold_labels, average="macro"
                )
                self._logger.debug(
                    "F1 score: %f", f1
                )
                if f1 > best_f1:
                    self._logger.debug(
                        "Best F1 score updated: %f", f1
                    )
                    best_f1 = f1
                    best_threshold = score_i
                    best_threshold_idx = i
            thresholds.append(best_threshold)
            prev_threshold = best_threshold
            prev_threshold_idx = min(best_threshold_idx + 1,
                                     len(score_space))
        return (best_f1, thresholds)

    def _preprocess(self, a_txt):
        """Overwrite parent's method lowercasing strings.

        Args:
          a_txt (str): text to be preprocessed

        Returns:
          str: preprocessed text

        """
        return super(LexiconBaseAnalyzer, self)._preprocess(a_txt).lower()

    def _split_polterm_matches(self, tags, term_matches):
        """Separate polterm matches according to their leading parts of speech.

        Args:
          tags (list[str]): PoS tags of tweet
          polterm_matches (list[tuple]): matches of polar terms

        Returns:
          PolTermMatches: matches separated by parts of speech

        """
        ret = self.PolTermMatches()
        for match_i in term_matches:
            score_i, start_i, end_i = match_i
            tags_i = set(tags[start_i:end_i + 1])
            for tags_j, contaier in ret.TAGS2CONTAINER:
                if tags_i & tags_j:
                    contaier.append(match_i)
                    break
            else:
                ret.adverbs.append(match_i)
        return ret


class CondProbLexiconBaseAnalyzer(LexiconBaseAnalyzer):
    """Abstract class for lexicon-based SA using conditional probabilities.

    """
    __metaclass__ = abc.ABCMeta

    def _read_lexicons(self, a_lexicons, a_encoding=ENCODING):
        """Overrides the method of the parent class, superseding.

        Args:
          a_lexicons (list): tags of the input instance
          a_encoding (str): input encoding

        Returns:
          cgsa.utils.trie.Trie: constructed polar terms trie

        """
        ret = Trie(a_ignorecase=True)
        for lexpath_i in a_lexicons:
            lexname = os.path.splitext(os.path.basename(
                lexpath_i
            ))[0]
            self._logger.debug(
                "Reading lexicon %s...", lexname
            )
            lexicon = pd.read_table(lexpath_i, header=None, names=LEX_CLMS,
                                    dtype=LEX_TYPES, encoding=a_encoding,
                                    error_bad_lines=False, warn_bad_lines=True,
                                    keep_default_na=False, na_values=[''],
                                    quoting=QUOTE_NONE)
            for i, row_i in lexicon.iterrows():
                term = USCORE_RE.sub(' ', row_i.term)
                polarity = row_i.polarity
                if NEG_SFX_RE.search(term):
                    # Taboada's method explicitly accounts for negations, so we
                    # skip negated entries from the lexicon altogether
                    continue
                term = self._preprocess(term)
                term_score = self._get_term_score(term, polarity, row_i.score)
                if term_score == 0.:
                    continue
                ret.add(SPACE_RE.split(term),
                        SPACE_RE.split(row_i.pos),
                        (lexname, polarity, term_score))
            self._logger.debug(
                "Lexicon %s read...", lexname
            )
        return ret

    def _get_term_score(self, term, polarity, lexicon_score):
        """Obtain term's score from conditional probabilities table.

        Args:
          term (str): term whose conditional probability should be checked
          polarity (str): polarity class of the term
          lexicon_score (float): sentiment score of the term from the lexicon
            (used as fallback)

        Returns:
          float: conditional probability of the term or its original
            lexicon score

        """
        if term in self._cond_probs:
            chck_term = term
        else:
            chck_term = term.lower()
            if chck_term not in self._cond_probs:
                return lexicon_score

        pos_prob, neg_prob = self._cond_probs[chck_term]
        if polarity == "positive":
            return pos_prob
        return -neg_prob
