#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from collections import defaultdict
from six import iteritems
import re

from cgsa.constants import PUNCT_RE, URI_RE
from cgsa.ml.base import ALEX, NEG_SFX_RE, MLBaseAnalyzer

##################################################################
# Variables and Constants
MAX_NGRAM_ORDER = 4
AT_RE = re.compile(r"\b@\w+", re.U)
ELONG_RE = re.compile(r"(\w)\1\1")
CAPS_RE = re.compile(r"^[A-ZÄÖÜ][A-ZÄÖÜ'0-9]*$")
CAPS = "%caps"
HSHTAG = "%hshtag"
UNI = "%uni"
BI = "%bi"
NONCONTIG = "%noncont"
AFF_CTXT = "affirmative"
NEG_CTXT = "negated"
ALL_POL = "all"
HSH_RE = re.compile(r"^#[\w\d_]+$")
SSPACE_RE = re.compile(r"\s+")
SPACE_RE = re.compile(r"\s\s+")
TAB_RE = re.compile(r" *\t *")
NEGATION_RE = re.compile(r" ^(?:nie(?:ma(?:nd|ls))?|nee|nö|ohne|"
                         r"nichts?|nirgendwo|kein(?:e?s|e[rmn])?)$ ",
                         re.I)
NEG_EMO_RE = re.compile(r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\(\[pP\{\|\\]+ # mouth
      |
      [\)\]/\}\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )""")
POS_EMO_RE = re.compile(r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\)\]dD/\}]+ # mouth
      |
      [\(\[dD\{@] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      |
      [*]+[-_]*[*]+
      |
      [-_]+[,.]+[-_]+
      |
      (?:&?lt;|<)3
    )""")
EXCL_MARK_RE = re.compile(r"!")
DBL_EXCL_MARK_RE = re.compile(r"!!")
DBL_EXCL_MARK_FEAT = "%ExclMarkCnt"
QMARK_RE = re.compile(r"\?")
DBL_QMARK_RE = re.compile(r"\?\?")
DBL_QMARK_FEAT = "%QMarkCnt"
LAST_QMARK_FEAT = "Last" + DBL_QMARK_FEAT
DBL_EXCL_MARK_FEAT = "%ExclMarkCnt"
LAST_EXCL_MARK_FEAT = "Last" + DBL_EXCL_MARK_FEAT
NEG_SFX = r"_NEG"
NEG_SFX_RE = re.compile(re.escape(re.escape(NEG_SFX)
                                  + r"(:?FIRST)?'"))
EXCL_QMARK_RE = re.compile(r"[!?][!?]")
EXCL_QMARK_FEAT = "%ExclQMarkCnt"
STAR_RE = re.compile(r"(?:\s+\*|\*\s+|\s+\*\s+)")


##################################################################
# Class
class MohammadAnalyzer(MLBaseAnalyzer):
    """Class for ML-based sentiment analysis.

    Attributes:

    """

    def __init__(self, *args, **kwargs):
        """Class constructor.

        Args:
          args (list): arguments
          kwargs (dict): additional keyword arguments

        """
        # initialize parent class
        super(MohammadAnalyzer, self).__init__(**kwargs)
        self.name = "mohammad"

    def _extract_feats(self, a_tweet):
        """Method for training the model.

        Args:
          a_tweet (cgsa.utils.data.Tweet):
            training instance to extract features from

        Returns:
          dict: extracted features and their values

        """
        self._logger.debug("tweet: %s", a_tweet)
        feats = {}
        toks = [self._preprocess(w.lemma) for w in a_tweet]
        text = ' '.join(toks)
        tags = [w.tag for w in a_tweet]
        # character n-grams
        self._extract_ngrams(feats, text, 3, 5, False)
        # number of capitalized words
        self._cnt(feats, toks, CAPS_RE, CAPS)
        # number of hashtags
        self._cnt(feats, toks, HSH_RE, HSHTAG)
        # number of words with duplicted characters
        self._cnt(feats, toks, AT_RE, "%at")
        # number of words with elongated characters
        self._cnt(feats, toks, ELONG_RE, "%elong")
        # tag statistics
        self._pos_stat(feats, tags)
        # punctuation statistics
        last_tok = toks[-1] if toks else ""
        self._punct_feats(feats, text, last_tok)
        # emoticon features
        self._emo_feats(feats, text, last_tok)
        # Brown cluster features
        # self._bc_feats(feats, toks)
        # apply negation and extract lexicon features
        feats["%NegCtxt"] = self._apply_negation(toks)
        # token n-grams
        ngrams = self._extract_ngrams(feats, toks,
                                      1, MAX_NGRAM_ORDER,
                                      True, tags)
        # lexicon features
        self._lex_feats(feats, ngrams, toks, tags)
        self._tertilize_feats(feats)
        self._logger.debug("feats: %r", feats)
        return feats

    def _apply_negation(self, a_toks):
        """Add negation suffix to tokens following negation particles.

        Args:
          a_toks (list[str]): list of input tokens

        Returns:
          int: number of negated contexts

        Note:
          modifies `a_toks` in place

        """
        neg_ctxt = 0
        neg_seen = False
        for i, itok in enumerate(a_toks):
            if PUNCT_RE.match(itok):
                neg_seen = False
            elif NEGATION_RE.search(itok):
                if not neg_seen:
                    neg_ctxt += 1
                neg_seen = True
            if neg_seen:
                a_toks[i] += NEG_SFX
        return neg_ctxt

    def _cnt(self, a_feats, a_input, a_re, a_feat_name):
        """Count occurrences of certain elements.

        Args:
          a_feats (dict): target dictionary of features to populate
          a_input (iterable): input sequence to extract the n-grams from
          a_re (re): regexp to match elements
          a_feat_name (str): name of the feature to create

        Returns:
          void:

        Note:
          modifies `a_feats` in-place

        """
        for itok in a_input:
            if a_re.search(itok):
                if a_feat_name in a_feats:
                    a_feats[a_feat_name] += 1
                else:
                    a_feats[a_feat_name] = 1

    def _pos_stat(self, a_feats, a_tags):
        """Count occurrence of tags.

        Args:
          a_feats (dict): target dictionary of features to populate
          a_tags (iterable): tags of the input sequence

        Returns:
          void:

        """
        for itag in a_tags:
            itag = "%PoS-" + itag
            if itag in a_feats:
                a_feats[itag] += 1
            else:
                a_feats[itag] = 1

    def _punct_feats(self, a_feats, a_txt, a_last_tok):
        """Count statistics on punctuation marks.

        Args:
          a_feats (dict): target dictionary of features to populate
          a_txt (str): text of the input sequence
          a_last_tok (str): last token of the input sequence

        Returns:
          void:

        """
        a_feats[DBL_EXCL_MARK_FEAT] = 0
        a_feats[DBL_QMARK_FEAT] = 0
        a_feats[EXCL_QMARK_FEAT] = 0
        # only beacause iteration costs less than creating a list
        for _ in DBL_EXCL_MARK_RE.finditer(a_txt):
            a_feats[DBL_EXCL_MARK_FEAT] += 1
        if EXCL_MARK_RE.search(a_last_tok):
            a_feats[LAST_EXCL_MARK_FEAT] = 1
        for _ in DBL_QMARK_RE.finditer(a_txt):
            a_feats[DBL_QMARK_FEAT] += 1
        if QMARK_RE.search(a_last_tok):
            a_feats[LAST_QMARK_FEAT] = 1
        for _ in EXCL_QMARK_RE.finditer(a_txt):
            a_feats[EXCL_QMARK_FEAT] += 1

    def _emo_feats(self, a_feats, a_txt, a_last_tok):
        """Add emoticon features.

        Args:
          a_feats (dict): target dictionary of features to populate
          a_txt (str): text of the input sequence
          a_last_tok (str): last token of the input sequence

        Returns:
          void:

        """
        if POS_EMO_RE.search(a_txt):
            a_feats["%PosEmo"] = 1
        if POS_EMO_RE.search(a_last_tok):
            a_feats["%PosEmoLast"] = 1
        if NEG_EMO_RE.search(a_txt):
            a_feats["%NegEmo"] = 1
        if NEG_EMO_RE.search(a_last_tok):
            a_feats["%NegEmoLast"] = 1

    def _lex_feats(self, a_feats, a_ngrams, a_toks, a_tags):
        """Extract manual lexicon features for the input.

        Args:
          a_feats (dict): target dictionary of features to populate
          a_ngrams (list[tuple]): list of extracted ngrams
          a_toks (list[str]): list of tweet's tokens
          a_tags (list[str]): list of tweet's tags

        Returns:
          void:

        Note:
          modifies `a_feats` in place

        """
        self._auto_lex_feats(a_feats, a_ngrams, a_toks, a_tags)
        self._mnl_lex_feats(a_feats, a_ngrams, a_toks, a_tags)

    def _auto_lex_feats(self, a_feats, a_ngrams, a_toks, a_tags):
        """Extract manual lexicon features for the input.

        Args:
          a_feats (dict): target dictionary of features to populate
          a_ngrams (list[tuple]): list of extracted ngrams
          a_toks (list[str]): list of tweet's tokens
          a_tags (list[str]): list of tweet's tags

        Returns:
          void:

        Note:
          modifies `a_feats` in place

        """
        keys = []
        max_pos = len(a_toks) - 1
        scores = defaultdict(float)
        for idcs, ngram, tags in a_ngrams:
            n = len(ngram)
            if n == 1:
                keys.append(UNI)
                if ngram[0].startswith('#'):
                    keys.append(HSHTAG)
                keys.append("%TAGS-" + '_'.join(tags))
                # Kiritchenko et al. (2014) do not use CAPS for automatic
                # lexicons
                # if CAPS_RE.match(ngram[0]):
                #     keys.append(CAPS)
            elif '*' in ngram:
                keys.append(NONCONTIG)
            elif n == 2:
                keys.append(BI)
            else:
                continue
            orig_ngram = ' '.join(ngram)
            orig_ngram = STAR_RE.sub("", orig_ngram)
            if NEG_SFX_RE.search(orig_ngram):
                orig_ngram = NEG_SFX_RE.sub("", orig_ngram)
                lex2chck = self._neg_term2auto
                ctxt = NEG_CTXT
            else:
                lex2chck = self._term2auto
                ctxt = AFF_CTXT
            if self._cs_fallback:
                if orig_ngram in lex2chck:
                    key2check = orig_ngram
                else:
                    key2check = CI_PRFX + orig_ngram.lower()
            else:
                key2check = orig_ngram.lower()
            if key2check in lex2chck:
                self._logger.debug("key2check %r found: %r",
                                   key2check, lex2chck[key2check])
                for ikey in keys:
                    for (lexname, pol), lex_scores \
                            in iteritems(lex2chck[key2check]):
                        lex_score_sum = sum(lex_scores)
                        lex_score_max = 0.
                        for s in lex_scores:
                            if abs(s) > lex_score_max:
                                lex_score_max = s
                        for ipol in (pol, ALL_POL):
                            feat_prfx = ALEX + "{lexname:s}_{pol:s}_{ctxt:s}" \
                                "_{key:s}".format(
                                    lexname=lexname, pol=ipol,
                                    ctxt=ctxt, key=ikey
                                )
                            feat_name = feat_prfx + "_cnt"
                            scores[feat_name] += len(lex_scores)
                            feat_name = feat_prfx + "_max"
                            if abs(lex_score_max) > abs(scores[feat_name]):
                                scores[feat_name] = lex_score_max
                            feat_name = feat_prfx + "_sum"
                            scores[feat_name] += lex_score_sum
                            feat_name = feat_prfx + "_avg"
                            scores[feat_name] += lex_score_sum \
                                / float(len(lex_scores))
                            if ipol != ALL_POL:
                                feat_name = feat_prfx + '_' \
                                    + SSPACE_RE.sub('_', key2check)
                                scores[feat_name] += lex_score_sum
                            pos = idcs[0]
                            if pos == max_pos:
                                feat_name = feat_prfx + "_last"
                                scores[feat_name] = lex_score_sum
            del keys[:]
        self._logger.debug("_auto_lex_feats: toks = %r", a_toks)
        self._logger.debug("_auto_lex_feats: tags = %r", a_tags)
        self._logger.debug("_auto_lex_feats: scores = %r", scores)
        a_feats.update(scores)

    def _mnl_lex_feats(self, a_feats, a_ngrams, a_toks, a_tags):
        """Extract manual lexicon features for the input.

        Args:
          a_feats (dict): target dictionary of features to populate
          a_ngrams (list[tuple]): list of extracted ngrams
          a_toks (list[str]): list of tweet's tokens
          a_tags (list[str]): list of tweet's tags

        Returns:
          void:

        Note:
          modifies `a_feats` in place

        """
        keys = []
        scores = defaultdict(float)
        for idcs, ngram, tags in a_ngrams:
            n = len(ngram)
            if n == 1:
                keys.append("tok")
                keys.append("%TAGS-" + '_'.join(tags))
                if ngram[0].startswith('#'):
                    keys.append(HSHTAG)
                if CAPS_RE.match(ngram[0]):
                    keys.append(CAPS)
            elif n == 2 or '*' in ngram:
                keys.append("tok")
            else:
                continue
            orig_ngram = ' '.join(ngram)
            orig_ngram = STAR_RE.sub("", orig_ngram).lower()
            if NEG_SFX_RE.search(orig_ngram):
                lex2chck = self._neg_term2mnl
                ctxt = NEG_CTXT
                orig_ngram = NEG_SFX_RE.sub("", orig_ngram)
            else:
                lex2chck = self._term2mnl
                ctxt = AFF_CTXT
            if orig_ngram in lex2chck:
                self._logger.debug("_mnl_lex_feats: lex2chck[%r] = %r",
                                   orig_ngram,
                                   lex2chck[orig_ngram])
                for (lexname, pol), lex_scores \
                        in iteritems(lex2chck[orig_ngram]):
                    for ikey in keys:
                        feat_name = '{lexname:s}_{pol:s}_{ctxt:s}_' \
                                    '{key:s}'.format(lexname=lexname,
                                                     pol=pol, ctxt=ctxt,
                                                     key=ikey)
                        scores[feat_name] += sum(lex_scores)
            del keys[:]
        self._logger.debug("_mnl_lex_feats: toks = %r", a_toks)
        self._logger.debug("_mnl_lex_feats: tags = %r", a_tags)
        self._logger.debug("_mnl_lex_feats: scores = %r", scores)
        a_feats.update(scores)
