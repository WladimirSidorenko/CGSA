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
from six import iteritems
import re

from cgsa.constants import KNOWN_LABELS

##################################################################
# Variables and Cosntants
BAR_RE = re.compile(r'|')
COLON_RE = re.compile(r'::+')
EQ_RE = re.compile(r'=')
SLASH_RE = re.compile(r'/')
SPACE_RE = re.compile(r' +')
TAB_RE = re.compile(r'\t')

MMAX = "mmax"


##################################################################
# Classes
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
            key, value = EQ_RE.split(feat_i)
            key_parts = COLON_RE.split(key)
            if len(key_parts) == 3:
                # parse MMAX feature
                markable, markable_id, attr = key_parts
                self._feats[MMAX][markable][markable_id][attr] = value
            else:
                self._feats[key] = value

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

    def __unicode__(self):
        """Return unicode representation of the given word.

        Returns:
          unicode:

        """
        if len(self._feats) == 1 and self._feats[MMAX]:
            return '_'
        mmax_feats = '|'.join([
            markable + "::" + markable_id + "::"
            + attr + '=' + value
            for markable in self.feats[MMAX]
            for markable_id in markable
            for attr, value in iteritems(markable_id)])
        feats = '|'.join([k + '-' + v
                          for k, v in iteritems(self.feats)
                          if k != MMAX])
        if feats:
            if mmax_feats:
                return feats + '|' + mmax_feats
            return feats
        return mmax_feats


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

    def __init__(self, form, lemma, tag, feats, deprel):
        """Class constructor.

        Attributes:
          form (str): word's form
          lemma (str): word's lemma
          tag (str): part-of-speech tag
          feats (str): word's features to be parsed
          deprel (str): dependency relation and index of the parent node

        """
        self.form = form
        self.lemma = lemma
        self.tag = tag
        self._parse_deps(deprel)
        self.feats = Features(feats)

    def __unicode__(self):
        """Return unicode representation of the given word.

        Returns:
          unicode:

        """
        ret = "form: '{:s}'; lemma: {:s}; tag: {:s}; deprel: {:s}/{:s}; " \
              " feats: {:s}".format(
                  self.form, self.lemma, self.tag,
                  unicode(self.prnt_idx + 1), self.deprel,
                  unicode(self.feats))
        return ret

    def _parse_deps(self, deprel):
        """Parse information about dependency relations.

        Args:
          deprel (str):

        """
        fields = SLASH_RE.split(deprel)
        self.prnt_idx = -1 if (fields[0] == '_' or fields[0] == '0') \
            else int(fields[0]) - 1
        self.deprel = fields[1]


class Tweet(object):
    """Class comprising relevant information about a tweet.

    Attributes:
      msg_id (str): id of the message
      words (list[str]): message's words with all their attributes
      dtree (list[str]): dependency tree

    """

    def __init__(self, tweet):
        """Class constructor.

        Attributes:
          tweet (str): tweet

        """
        self.msg_id = None
        self.label = None
        self.words = []
        self._parse(tweet)

    def __unicode__(self):
        """Return unicode representation of the given word.

        Returns:
          unicode:

        """
        # output the same string as the one that has been parsed
        ret = "{msg_id:s}\t{label:s}\t{tokens:s}\t{lemas:s}" \
              "\t{tags:s}\t{deps:s}\t{feats:s}".format(
                  self.msg_id, self.label,
                  ' '.join(w.form for w in self.words),
                  ' '.join(w.lemma for w in self.words),
                  ' '.join(w.tag for w in self.words),
                  ' '.join(str(w.prnt_idx + 1) + '/' + w.deprel
                           for w in self.words),
                  ' '.join(str(w.feats) for w in self.words)
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
        lemmas = SPACE_RE.split(fields[3])
        tags = SPACE_RE.split(fields[4])
        deps = SPACE_RE.split(fields[5])
        feats = SPACE_RE.split(fields[6])
        assert len(toks) == len(lemmas) == len(tags) \
            == len(deps) == len(feats), \
            "Unequal number of attributes at line {:r}"
        self.words = [
            Word(tok_i, lemma_i, tag_i, dep_i, feats_i)
            for tok_i, lemma_i, tag_i, dep_i, feats_i
            in zip(toks, lemmas, tags, deps, feats)
        ]
