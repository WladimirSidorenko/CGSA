#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from collections import defaultdict
from keras.models import Model
from keras.layers import Dense, Input
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
import abc
import numpy as np

from .base import DLBaseAnalyzer, EMB_INDICES_NAME, L2_COEFF
from .functional import FunctionalWord2Vec
from .layers import EMPTY_IDX


##################################################################
# Variables and Constants
DEP_LAYER_NAME = "dependencies"


##################################################################
# Class
class TreeRNNBaseAnalyzer(FunctionalWord2Vec, DLBaseAnalyzer):
    """Abstract base class for tree-RNNs.

    """
    DEP_PAD = [0, 0]

    def predict_proba(self, msg, yvec):
        wseq = self._tweet2wseq(msg)
        # self._logger.debug("deps: %r", deps)
        embs = np.array(
            self._pad(len(wseq), self._pad_value)
            + [self.get_test_w_emb(w) for w in wseq], dtype="int32")
        deps = [self.DEP_PAD] + self._get_deps(msg, len(msg) + 1)
        # self._logger.debug("embs: %r", embs)
        ret = self._model.predict([np.asarray([deps]),
                                   np.asarray([embs])],
                                  batch_size=1,
                                  verbose=2)
        yvec[:] = ret[0]

    def _digitize_data(self, train_x, dev_x):
        """Convert sequences of words to sequences of word indices.

        Args:
          train_x (list[list[str]]): training set
          dev_x (list[list[str]]): development set

        Returns:
          2-tuple[list, list]: digitized training and development sets

        """
        max_train_len = max(len(x) for x in train_x)
        max_dev_len = max(len(x) for x in dev_x) if dev_x else -1
        self._max_seq_len = max(max_train_len, max_dev_len) + 1
        self._min_width = self._max_seq_len
        # extract word embeddings
        train_embs, dev_embs = super(TreeRNNBaseAnalyzer, self)._digitize_data(
            train_x, dev_x
        )
        train_deps = pad_sequences([self._get_deps(x, self._max_seq_len)
                                    for x in train_x],
                                   value=self.DEP_PAD,
                                   maxlen=self._max_seq_len)
        train_x = [train_deps, train_embs]
        dev_deps = pad_sequences([self._get_deps(x, self._max_seq_len)
                                  for x in dev_x],
                                 value=self.DEP_PAD,
                                 maxlen=self._max_seq_len)
        dev_x = [dev_deps, dev_embs]
        return (train_x, dev_x)

    def _init_nn(self):
        """Initialize neural network.

        """
        self.init_w_emb()
        dependencies = Input(shape=(None, 2),
                             dtype="int32",
                             name=DEP_LAYER_NAME)
        emb_indices = Input(shape=(None,),
                            dtype="int32",
                            name=EMB_INDICES_NAME)
        embs = self.W_EMB(emb_indices)
        rnn = self._init_rnn([dependencies, embs])
        out = Dense(self._n_y,
                    activation="softmax",
                    kernel_regularizer=l2(L2_COEFF),
                    bias_regularizer=l2(L2_COEFF),
                    name="dense")(rnn)
        self._model = Model(inputs=[dependencies, emb_indices],
                            outputs=out)
        self._model.compile(**self._train_params)
        self._logger.debug(self._model.summary())

    @abc.abstractmethod
    def _init_rnn(self, *inputs):
        """Abstract method for defining a recurrent unit.

        Returns:
          keras.layer: custom keras layer

        """
        raise NotImplementedError

    def _get_deps(self, tweet, max_len):
        """Extract dependency indices of tweet tokens in topological order.

        Args:
          tweet (cgsa.utils.data.Tweet): analyzed input tweet
          max_len (int): maximum tweet length used for padding

        Returns:
          list[tuple]: list of dependencies as 2-tuples of the form
            (child_inex, parent_index)

        """
        deps = defaultdict(list)
        for i, tok_i in enumerate(tweet):
            deps[tok_i.prnt_idx].append(i)
        assert -1 in deps, \
            "No root found in tweet {:s}".format(str(tweet))
        ret = []
        active_nodes = [-1]
        offset = max_len - len(tweet)
        while active_nodes:
            prnt = active_nodes.pop(0)
            children = deps[prnt]
            active_nodes.extend(children)
            for child_i in children:
                ret.append((child_i + offset,
                            prnt + offset if prnt >= 0 else 0))
        assert len(ret) == len(tweet), \
            "Unmatching number of tokens and dependencies."
        ret.reverse()           # we proceed bottom-up
        ret = np.array(ret, dtype="int32")
        return ret

    def _pad(self, xlen, pad_value=EMPTY_IDX):
        """Return empty sequence.

        Args:
          xlen (int): length of the input instance (IGNORED)
          pad_value (int): value to use for padding

        Return:
          list: empty sequence

        """
        return [pad_value]

    def _pad_sequences(self, x):
        """Make all input instances of equal length.

        Args:
          x (list[np.array]): list of embedding indices

        Returns:
          x: list of embedding indices of equal lengths

        """
        return pad_sequences(x, maxlen=self._max_seq_len)

    def _tweet2wseq(self, msg):
        """Convert tweet to a sequence of word lemmas.

        Args:
          msg (cgsa.data.Tweet): input message

        Return:
          list: lemmas of informative words

        """
        return [w.lemma for w in msg]
