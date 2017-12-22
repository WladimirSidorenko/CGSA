#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from bisect import bisect_left
from collections import defaultdict
from six import iteritems
from six.moves import xrange
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import numpy as np
import re

from cgsa.ml.base import ALEX, MLBaseAnalyzer

##################################################################
# Variables and Constants
DFLT_C = 0.01
DFLT_CLS_WGHT = None
DFLT_PARAMS = {"class_weight": DFLT_CLS_WGHT, "loss": "hinge",
               "penalty": "l1", "dual": True,
               "multi_class": "crammer_singer"}
PARAM_GRID = {"clf__C": np.linspace(1e-2, 1, 7)}
MAX_NGRAM_ORDER = 4
WLDCARD = '*'
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
PUNCT_RE = re.compile(r"^[.,:;!?]+$")
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
NEG_RE_SFX = re.compile(r"_NEG", re.I)
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


##################################################################
# Class
class MohammadAnalyzer(MLBaseAnalyzer):
    """Class for ML-based sentiment analysis.

    Attributes:

    """

    def __init__(self, lexicons, a_clf=None, **kwargs):
        """Class constructor.

        Args:
          lexicons (list[str]): list of sentiment lexicons
          a_clf (None or Classifier Instance): classifier to use (None for
            default)
          kwargs (dict): additional keyword arguments

        """
        # initialize parent class
        super(MohammadAnalyzer, self).__init__(**kwargs)
        self.name = "mohammad"
        # read lexicons
        self._term2mnl = defaultdict(dict)
        self._neg_term2mnl = defaultdict(dict)
        self._term2auto = defaultdict(dict)
        self._neg_term2auto = defaultdict(dict)
        self._read_lexicons({"manual": (self._term2mnl, self._neg_term2mnl),
                             "auto": (self._term2auto, self._neg_term2auto)},
                            lexicons)
        # set up classifier
        clf = a_clf or LinearSVC(C=DFLT_C, **DFLT_PARAMS)
        self._model = Pipeline([("vect", DictVectorizer()),
                                ("clf", clf)])
        self.PARAM_GRID = PARAM_GRID
        self._cs_fallback = False
        self._feats2tertiles = {}

    def predict_proba(self, msg, yvec):
        feats = self._extract_feats(msg)
        dec = self._model.decision_function(feats)
        if len(dec.shape) > 1:
            dec = np.mean(dec, axis=0)
        for i, ival in enumerate(dec):
            yvec[self.classes_[i]] += ival

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

    def _extract_ngrams(self, a_feats, a_input, a_min_len, a_max_len,
                        a_skip_grams=False, tags=None):
        """Extract n-grams up to length n from iterable.

        Args:
          a_feats (dict): target dictionary of features to populate
          a_input (iterable): input sequence to extract the n-grams from
          a_min_len (int): minimum length of an n-gram
          a_max_len (int): maximum length of an n-gram
          a_skip_grams (bool): create skip grams (replace one of the
            tokens with wildcard)
          a_tags (list or None): optional list of part-of-speech tags

        Returns:
          set: extracted n-grams

        """
        ret = set()
        last_toks = []
        mmax = a_max_len + 1
        for i, x in enumerate(a_input):
            last_toks.append((i, x))
            if len(last_toks) > a_max_len:
                last_toks.pop(0)
            for i in xrange(a_min_len, mmax):
                ngram = last_toks[-i:]
                ret.add(tuple(ngram))
                if a_skip_grams and i > 1:
                    for j in xrange(len(ngram)):
                        tmp = ngram[j]
                        ngram[j] = (tmp[0], WLDCARD)
                        ret.add(tuple(ngram))
                        ngram[j] = tmp
        for ng in ret:
            a_feats['-'.join(n[-1] for n in ng)] = 1
        if tags:
            new_ret = set((tuple(n[0] for n in ng),
                           tuple(n[1] for n in ng),
                           tuple(tags[n[0]] for n in ng))
                          for ng in ret)
        else:
            new_ret = set((tuple(n[0] for n in ng),
                           tuple(n[1] for n in ng),
                           tuple())
                          for ng in ret)
        return new_ret

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
                # Kiritchenko et al., 2014, do not use CAPS for automatic
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
            if NEG_RE_SFX.search(orig_ngram):
                orig_ngram = NEG_RE_SFX.sub("", orig_ngram)
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
            if NEG_RE_SFX.search(orig_ngram):
                lex2chck = self._neg_term2mnl
                ctxt = NEG_CTXT
                orig_ngram = NEG_RE_SFX.sub("", orig_ngram)
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

    def _get_tertile(self, val, tertiles):
        """Determine the tertile of the given value.

        Args:
          val (float): feature value to determine the tertile for
          tertiles (np.array): array of tertile marks

        Returns:
          int: tertile of the given feature

        """
        return max(0, bisect_left(tertiles, val) - 1)

    def _tertilize_feats(self, feats):
        """Split specific features into separate attributes by their tertiles.

        Args:
          feats (dict): original features to be split

        Returns:
          void:

        Note:
          modifies `feats` in-place

        """
        for feat_name, tertiles in iteritems(self._feats2tertiles):
            if feat_name in feats:
                val = feats.pop(feat_name)
                tertile = self._get_tertile(val, tertiles)
                feats[feat_name + '-' + str(tertile)] = 1  # val
                modified = True
