#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from keras.models import Sequential
from keras.layers import (Bidirectional, Dense, Dropout, Embedding,
                          LSTM, GaussianNoise)
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
import numpy as np

from .base import DLBaseAnalyzer, L2_COEFF
from .layers import Attention


##################################################################
# Variables and Constants


##################################################################
# Class
class BaziotisAnalyzer(DLBaseAnalyzer):
    """Class for DL-based sentiment analysis.

    Attributes:

    """
    def __init__(self, *args, **kwargs):
        super(BaziotisAnalyzer, self).__init__(*args, **kwargs)
        self.name = "baziotis"
        self._max_seq_len = -1
        # Throughout this classifier, we use the same hyper-parameters as the
        # ones used by Baziotis et al. in their original work.
        # self.ndim = 300

    def _init_nn(self):
        self.init_w_emb()
        self._model = Sequential()
        # add embedding layer
        self._model.add(self.W_EMB)
        self._model.add(GaussianNoise(0.2))
        self._model.add(Dropout(0.3))
        # add two BiLSTM layers
        for _ in range(2):
            self._model.add(
                Bidirectional(
                    LSTM(150, recurrent_dropout=0.25,
                         return_sequences=True,
                         activity_regularizer=l2(L2_COEFF),
                         kernel_regularizer=l2(L2_COEFF),
                         bias_regularizer=l2(L2_COEFF),
                         recurrent_regularizer=l2(L2_COEFF))
                )
            )
            self._model.add(Dropout(0.5))
        # add Attention layer
        self._model.add(Attention(bias=True))
        self._model.add(Dropout(0.5))
        # add the final dense layer
        self._model.add(Dense(self._n_y,
                              activation="softmax",
                              activity_regularizer=l2(L2_COEFF),
                              kernel_regularizer=l2(L2_COEFF),
                              bias_regularizer=l2(L2_COEFF)))
        self._model.compile(optimizer="adadelta",
                            metrics=["categorical_accuracy"],
                            loss="categorical_hinge")

    def train(self, train_x, train_y, dev_x, dev_y, a_grid_search):
        max_train_len = max(len(x) for x in train_x)
        max_dev_len = max(len(x) for x in dev_x) if dev_x else -1
        self._max_seq_len = max(max_train_len, max_dev_len)
        self._min_width = self._max_seq_len
        super(BaziotisAnalyzer, self).train(train_x, train_y,
                                            dev_x, dev_y, a_grid_search)

    def predict_proba(self, msg, yvec):
        wseq = self._tweet2wseq(msg)
        embs = np.array(
            [self.get_test_w_emb(w) for w in wseq]
            + self._pad(len(wseq), self._pad_value), dtype="int32")
        self._logger.debug("embs: %r", embs)
        ret = self._model.predict(np.asarray([embs]),
                                  batch_size=1,
                                  verbose=2)
        yvec[:] = ret[0]

    def _init_w_emb(self):
        """Initialize task-specific word embeddings.

        """
        self.W_EMB = Embedding(len(self._w2i), self.ndim,
                               embeddings_initializer="he_normal",
                               embeddings_regularizer=l2(L2_COEFF),
                               input_length=self._max_seq_len)

    def _pad_sequences(self, x):
        """Make all input instances of equal length.

        Args:
          x (list[np.array]): list of embedding indices

        Returns:
          x: list of embedding indices of equal lengths

        """
        return pad_sequences(x, maxlen=self._max_seq_len)
