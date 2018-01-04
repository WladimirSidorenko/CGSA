#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

##################################################################
# Variables and Constants
from cgsa.ml.base import MLBaseAnalyzer
from cgsa.utils.dgtree import DGTree


##################################################################
# Class
class GamonAnalyzer(MLBaseAnalyzer):
    def __init__(self, *args, **kwargs):
        """Class constructor.

        Args:
          args (list): positional arguments
          kwargs (dict): additional keyword arguments

        """
        super(GamonAnalyzer, self).__init__(*args, **kwargs)
        self.name = "gamon"

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
        tags = [w.tag for w in a_tweet]
        deps = [(w.prnt_idx, w.deprel) for w in a_tweet.words]
        dgtree = DGTree(deps=deps)
        # part-of-speech trigrams
        # self._extract_ngrams(feats, tags, a_min_len=3, a_max_len=3)
        # constituent specific length measures (length of sentence, clauses,
        # adverbial/adjectival phrases, and noun phrases)
        self._get_constituent_lengths(feats, dgtree, a_tweet)
        # constituent structure in the form of context free phrase structure
        # patterns for each constituent in a parse tree. Example: DECL::NP VERB
        # NP (a declarative sentence consisting of a noun phrase a verbal head
        # and a second noun phrase)
        self._get_constituents(feats, dgtree, a_tweet)
        # Part of speech information coupled with semantic relations
        # (e.g. ``Verb Subject - Noun'' indicating a nominal subject to a
        # verbal predicate)
        # self._get_pos_constituents(feats, dgtree, a_tweet)
        # Logical form features provided by NLPWin, such as transitivity of a
        # predicate (-), tense information etc.
        for word_i in a_tweet:
            if word_i.tag.startswith('V') and "tense" in word_i.feats:
                self._logger.debug("tag: %r", word_i.tag)
                self._logger.debug("feats: %r", word_i.feats._feats)
                feats["tense" + '%' + word_i.feats["tense"]] = 1
        return feats

    def _get_constituent_lengths(self, feats, dgtree, tweet):
        """Add lengths of dg nodes to the feature set.

        Args:
          feats (dict): features to populate
          dgtree (cgsa.utils.dgtree.DGTree): dependency tree
          words (list[cgsa.utils.data.Tweet): analyzed tweet

        """
        nodes = [dgtree]
        # BFS over the tree
        while nodes:
            node_i = nodes.pop(0)
            if node_i.i >= 0:
                feat_name = node_i.deprel + '-' + tweet[node_i.i].tag
                if feat_name in feats:
                    feats[feat_name] = max(len(node_i), feats[feat_name])
                else:
                    feats[feat_name] = len(node_i)
            nodes.extend(node_i.children)

    def _get_constituents(self, feats, dgtree, tweet):
        """Add constituents to the feature set.

        Args:
          feats (dict): features to populate
          dgtree (cgsa.utils.dgtree.DGTree): dependency tree
          words (list[cgsa.utils.data.Tweet): analyzed tweet

        """
        nodes = [dgtree]
        # BFS over the tree
        while nodes:
            node_i = nodes.pop(0)
            if node_i.i >= 0:
                feat_name = node_i.deprel + '-' + tweet[node_i.i].tag + '|'
                for child_j in node_i.children:
                    feat_name += child_j.deprel + '-' \
                                 + tweet[child_j.i].tag + '|'
                feats[feat_name] = 1
            nodes.extend(node_i.children)

    def _get_pos_constituents(self, feats, dgtree, tweet):
        """Add tuples of the form (parent PoS - Connector - child PoS) to features.

        Args:
          feats (dict): features to populate
          dgtree (cgsa.utils.dgtree.DGTree): dependency tree
          words (list[cgsa.utils.data.Tweet): analyzed tweet

        """
        nodes = [dgtree]
        # BFS over the tree
        while nodes:
            node_i = nodes.pop(0)
            if node_i.i >= 0:
                prnt_name = tweet[node_i.i].tag[:1]
                for child_j in node_i.children:
                    child_name = tweet[child_j.i].tag[:1]
                    feats[prnt_name
                          + '-' + child_j.deprel
                          + '-' + child_name] = 1
            nodes.extend(node_i.children)
