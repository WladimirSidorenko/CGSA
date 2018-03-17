#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from copy import deepcopy
from lime.lime_text import LimeTextExplainer, TextDomainMapper
from lime import explanation
import itertools
import numpy as np

from cgsa.utils.common import is_relevant
from cgsa.utils.data import Features


##################################################################
# Classes
class IndexedString(object):
    """String with various indexes."""

    def __init__(self, msg, split_expression=r'\W+', bow=True):
        """Initializer.

        Args:
            msg: message containing tweet text
            split_expression: string will be split by this (IGNORED)
            bow: if True, a word is the same everywhere in the text - i.e. we
                 will index multiple occurrences of the same word. If False,
                 order matters, so that the same word will have different ids
                 according to position.
        """
        self.msg = msg
        self.inverse_vocab = []
        self.positions = []
        self.bow = bow
        vocab = {}
        non_vocab = set()
        for i, word in enumerate(self.msg):
            lemma = word.lemma
            if lemma in non_vocab:
                continue
            if not is_relevant(lemma):
                non_vocab.add(lemma)
                continue
            if bow:
                if lemma not in vocab:
                    vocab[lemma] = len(vocab)
                    self.inverse_vocab.append(lemma)
                    self.positions.append([])
                idx_word = vocab[lemma]
                self.positions[idx_word].append(i)
            else:
                self.inverse_vocab.append(lemma)
                self.positions.append(i)
        self.positions = np.array(self.positions)

    def raw_string(self):
        """Returns the original raw string"""
        return self.msg

    def num_words(self):
        """Returns the number of tokens in the vocabulary for this document."""
        return len(self.inverse_vocab)

    def word(self, id_):
        """Returns the word that corresponds to id_ (int)"""
        return self.inverse_vocab[id_]

    def string_position(self, id_):
        """Returns a np array with indices to id_ (int) occurrences"""
        return self.positions[id_]

    def inverse_removing(self, words_to_remove):
        """Returns modified tweet after removing appropriate words.

        If self.bow is false, replaces word with UNKWORDZ instead of removing
        it.

        Args:
            words_to_remove: list of ids (ints) to remove

        Returns:
            original raw string with appropriate words removed.
        """
        def delete_word(msg, i):
            word_i = msg.words[i]
            prnt_idx = word_i.prnt_idx
            if prnt_idx >= 0:
                prnt = msg.words[prnt_idx]
                prnt.children.remove(i)
                prnt.children.extend(word_i.children)
            for child_j in self.words.children:
                child_j.prnt_idx = prnt_idx
            msg.words.pop(i)

        mask = np.ones(len(self.msg), dtype='bool')
        mask[self.__get_idxs(words_to_remove)] = False

        msg_logger = self.msg._logger
        word_loggers = [w._logger for w in self.msg]
        self.msg._logger = None
        for w in self.msg:
            w._logger = None
        msg = deepcopy(self.msg)
        self.msg._logger = msg._logger = msg_logger
        for w1, w2, w_logger in zip(self.msg, msg, word_loggers):
                w1._logger = w2._logger = w_logger

        for i in range(len(self.msg) - 1, -1, -1):
            if not mask[i]:
                if self.bow:
                    msg.words[i].lemma = 'UNKWORDZ'
                    msg.words[i].form = 'UNKWORDZ'
                    msg.words[i].tag = 'NP'
                    msg.words[i].features = Features('_')
                else:
                    delete_word(msg, i)
        # print("msg before removal: %s", self.msg)
        # print("msg after removal: %s", msg)
        return msg

    def __get_idxs(self, words):
        """Returns indexes to appropriate words."""
        if self.bow:
            return list(itertools.chain.from_iterable(
                [self.positions[z] for z in words]))
        else:
            return self.positions[words]


class Explainer(LimeTextExplainer):
    def explain_instance(self,
                         text_instance,
                         classifier_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='cosine',
                         model_regressor=None):
        """This basically just a copy of :class:`LimeTextExplainer` with our custom
           implementation of :class:`IndexedString`.
        """
        indexed_string = IndexedString(text_instance, bow=self.bow,
                                       split_expression=self.split_expression)
        domain_mapper = TextDomainMapper(indexed_string)
        data, yss, distances = self.__data_labels_distances(
            indexed_string, classifier_fn, num_samples,
            distance_metric=distance_metric)
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names,
                                          random_state=self.random_state)
        ret_exp.predict_proba = yss[0]
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                data, yss, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def __data_labels_distances(self, *args, **kwargs):
        return self._LimeTextExplainer__data_labels_distances(*args, **kwargs)
