#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module for reading and outputting data.

Attributes:


"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from collections import defaultdict
from future.utils import python_2_unicode_compatible
from six import iteritems
import re

from cgsa.utils.common import LOGGER
from cgsa.constants import KNOWN_LABELS

##################################################################
# Variables and Constants
BAR_RE = re.compile(r'[|]')
COLON_RE = re.compile(r'::+')
EQ_RE = re.compile(r'=')
SLASH_RE = re.compile(r'/')
SPACE_RE = re.compile(r' +')
TAB_RE = re.compile(r'\t')

MMAX = "mmax"
# compatibility with newer mate versions
VAL2KEY = {"1": "person", "2": "person", "3": "person",
           "nom": "case", "gen": "case", "dat": "case",
           "acc": "case", "sg": "number", "pl": "number",
           "pres": "tense"}


##################################################################
# Classes
@python_2_unicode_compatible
class Features(object):
    """Class comprising relevant information about features.

    """
    def __init__(self, feats):
        """Parse information about dependency relations.

        Args:
          feats (str): features to parse

        Retuns:
          void:

        """
        self._feats = {MMAX: defaultdict(
            lambda: defaultdict(dict))}
        if feats == '_':
            return
        for feat_i in BAR_RE.split(feats):
            if EQ_RE.search(feat_i):
                key, value = EQ_RE.split(feat_i)
            else:
                value = feat_i
                key = VAL2KEY.get(value, "UNKFEAT")
            if value.lower() == "true" or value.lower() == "false":
                value = bool(value)
            key_parts = COLON_RE.split(key)
            if len(key_parts) == 3:
                # parse MMAX feature
                markable, markable_id, attr = key_parts
                self._feats[MMAX][markable][markable_id][attr] = value
            else:
                self._feats[key] = value

    def get(self, key):
        """Return feaure's value if it is present.

        Args:
          key (str): feature's name

        """
        return self._feats.get(key)

    def __contains__(self, key):
        """Check feaure's presence.

        Args:
          key (str): feature's name

        """
        return key in self._feats

    def __getitem__(self, key):
        """Return feaure's value.

        Args:
          key (str): feature's name

        """
        return self._feats[key]

    def __haskey__(self, key):
        """Check feaure's presence.

        Args:
          key (str): feature's name

        """
        return key in self._feats

    def __setitem__(self, key, value):
        """Set feaure's value.

        Args:
          key (str): feature's name
          value (object): new feature's value

        """
        self._feats[key] = value
        return value

    def __str__(self):
        """Return unicode representation of the given word.

        Returns:
          unicode:

        """
        if len(self._feats) == 1 and self._feats[MMAX]:
            return '_'
        mmax_feats = '|'.join([
            mkbl + "::" + markable_id + "::"
            + attr + '=' + str(value)
            for mkbl, mkbl_val in iteritems(self._feats[MMAX])
            for markable_id in mkbl_val
            for attr, value in iteritems(mkbl_val[markable_id])])
        feats = '|'.join([k + '-' + str(v)
                          for k, v in iteritems(self._feats)
                          if k != MMAX])
        if feats:
            if mmax_feats:
                return feats + '|' + mmax_feats
            return feats
        elif mmax_feats:
            return mmax_feats
        else:
            return '_'


@python_2_unicode_compatible
class Word(object):
    """Class comprising relevant information about a word.

    Attributes:
      form (str): word's form
      lemma (str): word's lemma
      tag (str): part-of-speech tag
      feats (dict): word's features
      deprel (str): dependency relation connecting the word to its parent
      prnt_idx (int or None): index of the parent node

    """

    def __init__(self, form, lemma, tag, deprel, feats):
        """Class constructor.

        Attributes:
          form (str): word's form
          lemma (str): word's lemma
          tag (str): part-of-speech tag
          deprel (str): dependency relation and index of the parent node
          feats (str): word's features to be parsed

        """
        self._logger = LOGGER
        self.form = form
        self.lemma = lemma
        self.tag = tag
        self._parse_deps(deprel)
        self.feats = Features(feats)
        self.children = []

    def __str__(self):
        """Return unicode representation of the given word.

        Returns:
          unicode:

        """
        ret = "form: '{:s}'; lemma: {:s}; tag: {:s}; deprel: {:s}/{:s}; " \
              " feats: {:s}".format(
                  self.form, self.lemma, self.tag,
                  str(self.prnt_idx + 1), self.deprel,
                  str(self.feats))
        return ret

    def _parse_deps(self, deprel):
        """Parse information about dependency relations.

        Args:
          deprel (str):

        """
        fields = SLASH_RE.split(deprel)
        if fields[0] == '_' or fields[0] == '0':
            self.prnt_idx = -1
            self.deprel = '_'
        else:
            self.prnt_idx = int(fields[0]) - 1
            self.deprel = fields[1]


@python_2_unicode_compatible
class Tweet(object):
    """Class comprising relevant information about a tweet.

    Attributes:
      msg_id (str): id of the message
      words (list[str]): message's words with all their attributes
      dtree (list[str]): dependency tree

    """

    def __init__(self, tweet):
        """Class constructor.

        Args:
          tweet (str): tweet

        """
        self.msg_id = None
        self.label = None
        self.words = []
        self.iwords = iter(self.words)
        self._logger = LOGGER
        self._parse(tweet)

    def __iter__(self):
        """Private method required for iteration.

        """
        self.iwords = iter(self.words)
        return self.iwords

    def __len__(self):
        """Return number of tokens in the tweet.

        """
        return len(self.words)

    def __getitem__(self, key):
        """Private method required for iteration.

        """
        return self.words[key]

    def __next__(self):
        """Private method required for iteration.

        """
        return next(self.iwords)

    def __str__(self):
        """Return unicode representation of the given word.

        Returns:
          unicode:

        """
        # output the same string as the one that has been parsed
        ret = "{msg_id:s}\t{label:s}\t{tokens:s}\t{lemmas:s}" \
              "\t{tags:s}\t{deps:s}\t{feats:s}".format(
                  msg_id=self.msg_id, label=self.label,
                  tokens=' '.join(w.form for w in self.words),
                  lemmas=' '.join(w.lemma for w in self.words),
                  tags=' '.join(w.tag for w in self.words),
                  deps=' '.join(str(w.prnt_idx + 1) + '/' + w.deprel
                                for w in self.words),
                  feats=' '.join(str(w.feats) for w in self.words)
              )
        return ret

    def _parse(self, tweet):
        """Parse tweet's string.

        Attributes:
          tweet (str): tweet to be parse

        """
        fields = TAB_RE.split(tweet)
        assert len(fields) == 7, \
            "Incorrect line format (expected seven tab fields): {!r}".format(
                tweet
            )
        self.msg_id = fields[0]
        assert fields[1] in KNOWN_LABELS, \
            "Unknown sentiment label: {!r}".format(fields[1])
        self.label = fields[1]
        toks = SPACE_RE.split(fields[2])
        lemmas = SPACE_RE.split(fields[3].lower())
        tags = SPACE_RE.split(fields[4])
        deps = SPACE_RE.split(fields[5])
        feats = SPACE_RE.split(fields[6])
        n = len(toks)
        if n != len(lemmas) or n != len(tags) \
           or n != len(deps) or n != len(feats):
            self._logger.error("Tokens: %d", n)
            self._logger.error("Lemmas: %d", len(lemmas))
            self._logger.error("Tags: %d", len(tags))
            self._logger.error("Dependencies: %d", len(deps))
            self._logger.error("Features: ", len(feats))
            for tok_i, lemma_i in zip(toks, lemmas):
                if tok_i.lower() != lemma_i.lower():
                    self._logger.error("{!r} <-> {!r}".format(tok_i, lemma_i))
            assert False, \
                "Unequal number of attributes at line {!r}".format(
                    tweet)
        self.words = [
            Word(tok_i, lemma_i, tag_i, dep_i, feats_i)
            for tok_i, lemma_i, tag_i, dep_i, feats_i
            in zip(toks, lemmas, tags, deps, feats)
        ]
        for i, w_i in enumerate(self.words):
            prnt_idx = w_i.prnt_idx
            if prnt_idx < 0:
                continue
            prnt = self.words[prnt_idx]
            prnt.children.append(i)
