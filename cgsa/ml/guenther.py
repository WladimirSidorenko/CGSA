#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from collections import defaultdict
from copy import deepcopy
from six import iteritems, iterkeys
from six.moves import range
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import numpy as np
import re

from cgsa.constants import (EMOTICON_RE, PUNCT_RE,
                            POSITIVE, NEGATIVE,
                            URI_RE, WORD_RE)
from cgsa.ml.base import NEG_SFX_RE, MLBaseAnalyzer


##################################################################
# Variables and Constants
STOP_WORDS = set([
    "der", "die", "das", "dem", "den",
    "ein", "eine", "eines", "einer", "einem", "einen",
    "sein", "seine", "seines", "seiner", "seinem", "seinen",
    "mein", "meine", "meines", "meiner", "meinem", "meinen",
    "dein", "deine", "deines", "deiner", "deinem", "deinen",
    "ihr", "ihre", "ihres", "ihrer", "ihrem", "ihren",
    "habe", "hast", "hat", "haben", "habt",
    "hatte", "hattest", "hattet", "hatten", "hattet", "gehabt",
    "hätte", "hättest", "hättet", "hätten", "hättet",
    "bin", "bist", "ist", "sind", "seid",
    "war", "warst", "war", "waren", "wart",
    "wär", "wärest", "wäre", "wären", "wäret",
    "werde", "wirst", "wird", "werden",
    "wurde", "wurderst", "wurdet", "wurden",
    "würde", "würderst", "würdet", "würden"
])
HSHTAG_RE = re.compile(r"\b#\w")
TRIPLE_CHAR = re.compile(r"(?P<char>\w)(?P=char)(?P=char)")
N_REPLICAS = 10
SUBSAMPLE_PROB = 0.2
USE_SUBSAMPLING = True
DFLT_ALPHA = 1e-4
DFLT_L1_RATIO = 1e-2


##################################################################
# Class
class GuentherAnalyzer(MLBaseAnalyzer):
    @staticmethod
    def check_tok(tok):
        """Check whether given token should be considered for analysis.

        Args:
          tok (str): token to check

        Returns:
          bool: True if token is uninformative, False otherwise

        """
        tok = NEG_SFX_RE.sub("", tok)
        return (not PUNCT_RE.match(tok)
                and tok.lower() not in STOP_WORDS)

    """Class for ML-based sentiment analysis.

    Attributes:

    """

    def __init__(self, lexicons, a_clf=None, **kwargs):
        """Class constructor.

        Args:
          kwargs (dict): additional keyword arguments

        """
        super(GuentherAnalyzer, self).__init__([], a_clf, **kwargs)
        self.name = "guenther"
        # read lexicons
        self._term2pol_score = defaultdict(dict)
        self._neg_term2pol_score = defaultdict(dict)
        self._read_lexicons({"any": (self._term2pol_score,
                                     self._neg_term2pol_score)},
                            lexicons)
        # set up classifier
        clf = a_clf or SGDClassifier(penalty="elasticnet",
                                     alpha=DFLT_ALPHA,
                                     l1_ratio=DFLT_L1_RATIO,
                                     n_iter=1000, class_weight="balanced")
        self._model = Pipeline([("vect", DictVectorizer()),
                                ("clf", clf)])
        self.PARAM_GRID = {"clf__alpha": np.linspace(1e-4, 1e-2, 5),
                           "clf__l1_ratio":  np.linspace(1e-2, 9e-1, 5)}

    def train(self, train_x, train_y, dev_x, dev_y, a_grid_search):
        if USE_SUBSAMPLING:
            train_x = [self._extract_feats(t) for t in train_x]
            self._logger.info("Subsampling %d training examples.",
                              len(train_x))
            self._subsample(train_x, N_REPLICAS, SUBSAMPLE_PROB)
            self._logger.info("Subsampled %d training examples.",
                              len(train_x))
            train_y *= N_REPLICAS + 1
            self._subsample(dev_x, N_REPLICAS, SUBSAMPLE_PROB)
            dev_x = [self._extract_feats(t) for t in dev_x]
        super(GuentherAnalyzer, self).train(train_x, train_y,
                                            dev_x, dev_y, a_grid_search,
                                            a_extract_feats=False)

    def _extract_feats(self, a_tweet):
        """Method for training the model.

        Args:
          a_tweet (cgsa.utils.data.Tweet):
            training instance to extract features from

        Returns:
          dict: extracted features and their values

        """
        # self._logger.debug("tweet: %s", a_tweet)
        feats = {}
        forms = [self._preprocess(w.form) for w in a_tweet]
        lemmas = [self._preprocess(w.lemma) for w in a_tweet]
        tags = [w.tag for w in a_tweet]
        match_input = [(f, l, t)
                       for f, l, t
                       in zip(forms, lemmas, tags)]
        # match negations
        neg_matches = [(start, end)
                       for _, start, end
                       in self._negations.search(match_input)]
        # self._logger.debug("neg_matches: %r", neg_matches)
        # match boundaries
        boundaries = self._find_boundaries(match_input)
        # self._logger.debug("boundaries: %r", boundaries)
        # determine indices of negated tokens
        negated_indices = self._find_negated_toks(
            neg_matches, boundaries
        )
        # self._logger.debug("negated_indices: %r", negated_indices)
        # negate forms and lemmas in negated context
        forms = self._negate(forms, negated_indices)
        pruned_forms = [form_i for form_i in
                        filter(self.check_tok, forms)]
        lemmas = self._negate(lemmas, negated_indices)
        # token n-grams
        self._logger.debug("forms: %r", forms)
        self._logger.debug("pruned_forms: %r", pruned_forms)
        self._logger.debug("lemmas: %r", lemmas)
        # unigrams
        ngrams = self._extract_ngrams(feats, forms, tags,
                                      forms, boundaries)
        self._logger.debug("ngrams (0): %r", ngrams)
        # bigrams
        ngrams |= super(GuentherAnalyzer, self)._extract_ngrams(
            feats, pruned_forms, a_min_len=2, a_max_len=2)
        self._logger.debug("ngrams (1): %r", ngrams)
        # lemma unigrams
        ngrams |= self._extract_ngrams(feats, lemmas, tags,
                                       forms, boundaries)
        self._logger.debug("ngrams (2): %r", ngrams)
        # URI feature
        text = ' '.join(forms)
        if URI_RE.search(text):
            feats["%URI-FEAT"] = 1
        # hashtag feature
        text = ' '.join(forms)
        if HSHTAG_RE.search(text):
            feats["%HSHTAG-FEAT"] = 1
        # lexicon features
        self._lex_feats(feats, ngrams)
        # hashtag feature
        self._logger.debug("feats: %r", feats)
        return feats

    def _extract_ngrams(self, a_feats, a_tokens, a_tags,
                        a_forms, a_boundaries):
        """Extract n-grams up to length n from iterable.

        Args:
          a_feats (dict): target dictionary of features to populate
          a_tokens (list[str]): sequence tokens
          a_tags (list): part-of-speech tags
          a_forms (list[str]): original token forms
          a_boundaries (list[int]): indices of clause boundaries

        Returns:
          set: extracted n-grams

        """
        ret = set()
        last_toks = []
        for i, (token_i, tag_i) in enumerate(zip(a_tokens, a_tags)):
            wght = 1.
            if WORD_RE.match(token_i) and token_i.upper() == token_i:
                wght += 1.
            if TRIPLE_CHAR.search(token_i):
                wght += 1.
            if tag_i.startswith("ADJ") or EMOTICON_RE.match(token_i):
                wght += 1.

            if '?' in self._get_sent_punct(i, a_forms, a_boundaries):
                wght /= 2.
            a_feats[token_i] = wght
            ret.add((None, token_i, None))
        return ret

    def _find_negated_toks(self, neg_matches, boundaries):
        """Find indices of negated tokens.

        Args:
          neg_matches (list[tuple]): start and end indices of negations
          boundaries (list[tuple]): start and end indices of boundaries

        Returns:
          set(int): indices of negated tokens

        """
        negated_toks = set(i
                           for neg_start, neg_end in neg_matches
                           for i in range(neg_start, neg_end + 1))
        ret = set()
        prev_start = 0
        for boundary_start, boundary_end in sorted(boundaries):
            toks = set(t for t in range(prev_start, boundary_end))
            intersection = toks & negated_toks
            self._logger.debug("intersection: %r", intersection)
            # Variant 1: we consider the whole segment as negated
            # if intersection:
            #     ret |= toks
            # Variant 2: we only consider as negated tokens between the
            # negation marker and the next boundary to the right
            if intersection:
                ret |= set(t
                           for t in range(
                                   sorted(intersection)[0] + 1,
                                   boundary_end))
            prev_start = boundary_end + 1
        return ret

    def _lex_feats(self, feats, ngrams):
        """Add lexicon features.

        Args:
          feats (dict): feature dictionary to be modified
          ngrams (set[tuple]): n-grams to check in lexicons

        Returns:
          None:

        Note:
          modifies `feats` in place

          For each of the three sentiment lexica two features capture whether
          the majority of the tokens in the message were in the positive or
          negative sentiment list.  The same is done for hashtags using the NRC
          hashtag sentiment lexicon (Mohammad et al., 2013).

        """
        lex_stat = defaultdict(lambda: {POSITIVE: 0, NEGATIVE: 0})

        def check_term(lex_stat, term, fallback_term,
                       lexicon, fallback_lexicon=None):

            def check(term, fallback_term, lexicon):
                if term in lexicon:
                    return iterkeys(lexicon[term])
                elif fallback_term and fallback_term in lexicon:
                    return iterkeys(lexicon[fallback_term])

            # first check in the primary lexicon
            lex_pol = check(term, fallback_term, lexicon)
            if lex_pol is not None:
                for (lex_name_i, pol_i) in lex_pol:
                    lex_stat[lex_name_i][pol_i] += 1

            # if fallback lexicon is specified, check the term there and
            # reverse its polarity if found
            if fallback_lexicon is not None:
                lex_pol = check(term, fallback_term, fallback_lexicon)
                if lex_pol is not None:
                    for (lex_name_i, pol_i) in lex_pol:
                        # reverse polarity
                        pol_i = POSITIVE if pol_i == NEGATIVE else POSITIVE
                        self._logger.debug("lex_stat[%s]: %r",
                                           lex_name_i, lex_stat[lex_name_i])
                        lex_stat[lex_name_i][pol_i] += 1

        for _, ngram, _ in ngrams:
            orig_ngram = ' '.join(ngram)
            if NEG_SFX_RE.search(orig_ngram):
                ngram = self._preprocess(
                    NEG_SFX_RE.sub("", orig_ngram)
                    )
                check_term(lex_stat, ngram, orig_ngram,
                           self._term2pol_score)
            else:
                ngram = self._preprocess(orig_ngram)
                check_term(lex_stat, ngram, orig_ngram,
                           self._neg_term2pol_score, self._term2pol_score)
        self._logger.debug("lex_stat: %r", lex_stat)
        for lex_name, pol_stat in iteritems(lex_stat):
            if pol_stat[POSITIVE] > pol_stat[NEGATIVE]:
                feats[lex_name + '%' + POSITIVE] = 1.
                feats[lex_name + '%' + NEGATIVE] = 0.
            elif pol_stat[POSITIVE] < pol_stat[NEGATIVE]:
                feats[lex_name + '%' + NEGATIVE] = 1.
                feats[lex_name + '%' + POSITIVE] = 0.

    def _negate(self, words, negated_indices):
        """Add negation suffix to words in negated context.

        Args:
          words (list[str]): words to be negated
          negated_indices (list[int]): indices of words that should be negated

        """
        return [w_i + "_NEG" if i in negated_indices else w_i
                for i, w_i in enumerate(words)]

    def _subsample(self, data, n_replicas, subsample_prob):
        """Repeat training examples, randomly subsampling features.

        Args:
          data (list[dict]): training instances
          n_replicas (int): number of replicas to generate from original data
          subsample_prob (float): probability of removing a ranom feature

        Returns:
          list[dict]: data

        """
        end = len(data)
        for n in range(n_replicas):
            for feats_i in data[:end]:
                feats = deepcopy(feats_i)
                keys = feats_i.keys()
                mask = np.random.binomial(1, subsample_prob, len(feats_i))
                for k_i, m_i in zip(keys, mask):
                    if m_i:
                        feats.pop(k_i)
                data.append(feats)
