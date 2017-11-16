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

from sklearn.metrics import f1_score
from six import iterkeys
import abc
import numpy as np
import pandas as pd
import os

from cgsa.constants import (IDX2CLS, INTENSIFIERS, PUNCT_RE,
                            SPACE_RE, USCORE_RE)
from cgsa.base import (BaseAnalyzer, ENCODING, LEX_CLMS, LEX_TYPES,
                       NEG_SFX_RE, QUOTE_NONE)
from cgsa.utils.common import LOGGER
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

    def __init__(self, lexicons=[]):
        """Class constructor.

        Args:
          lexicons (list[str]): list of sentiment lexicons

        """
        assert lexicons, \
            "Provide at least one lexicon for lexicon-based method."
        self._logger = LOGGER
        self._intensifiers = self._read_intensifiers(INTENSIFIERS)
        self._boundaries = self._words2trie(BOUNDARIES)
        self._negations = self._words2trie(NEGATIONS)
        self._polar_terms = self._read_lexicons(lexicons,
                                                a_encoding=ENCODING)

    def reset(self):
        """Remove members which cannot be serialized.

        """
        super(LexiconBaseAnalyzer, self).reset()
        self._logger = None
        self._intensifiers.reset()
        self._boundaries.reset()
        self._negations.reset()
        self._polar_terms.reset()

    def restore(self):
        """Restore members which could not be serialized.

        """
        self._logger = LOGGER
        self._intensifiers.restore()
        self._boundaries.restore()
        self._negations.restore()
        self._polar_terms.restore()

    @abc.abstractmethod
    def _compute_so(self, tweet):
        """Compute semantic orientation of a tweet.

        Args:
          tweet (cgsa.utils.data.Tweet): input message

        Returns:
          tuple: total SO score, total SO count, average SO value

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
        prev_threshold_idx = 0
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
                f1 = f1_score(predicted_labels, gold_labels,
                              average="macro")
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
            prev_threshold_idx = best_threshold_idx
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
