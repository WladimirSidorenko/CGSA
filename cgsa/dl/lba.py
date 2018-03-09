#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from csv import QUOTE_NONE
from collections import defaultdict
from keras.models import Model
from keras.layers import (Average, Dense, Dropout, Bidirectional, LSTM,
                          GaussianNoise, Input)
from keras.regularizers import l2
from six import iteritems
import numpy as np
import os
import pandas as pd

from cgsa.utils.common import normlex
from cgsa.base import (LEX_CLMS, LEX_TYPES, NEG_SFX_RE, USCORE_RE)
from cgsa.constants import ENCODING

from .base import (EMB_INDICES_NAME, EMPTY_TOK, UNK_TOK, L2_COEFF)
from .baziotis import BaziotisAnalyzer
from .layers import Attention, LBA, EMPTY_IDX, UNK_IDX


##################################################################
# Variables and Constants
LEX_INDICES_NAME = "lexicon_indices"


##################################################################
# Class
class LBAAnalyzer(BaziotisAnalyzer):
    """Class for DL-based sentiment analysis.

    Attributes:

    """
    def __init__(self, lexicons, *args, **kwargs):
        assert lexicons, \
            "Provide at least one lexicon for the LBA method."
        super(LBAAnalyzer, self).__init__(*args, **kwargs)
        self.name = "lba"
        self._read_lexicons(None, lexicons)

    def predict_proba(self, msg, yvec):
        wseq = self._tweet2wseq(msg)
        embs = np.array(
            [self.get_test_w_emb(w) for w in wseq]
            + self._pad(len(wseq), self._pad_value), dtype="int32")
        lex_embs = [wseq]
        self._wseq2emb_ids(lex_embs, self.get_lex_emb_i)
        ret = self._model.predict([np.asarray([embs]),
                                   np.asarray(lex_embs)],
                                  batch_size=1, verbose=2)
        yvec[:] = ret[0]

    def _init_nn(self):
        self.init_w_emb()
        emb_indices = Input(shape=(None,),
                            dtype="int32",
                            name=EMB_INDICES_NAME)
        lex_indices = Input(shape=(None,),
                            dtype="int32",
                            name=LEX_INDICES_NAME)
        vec_embs = self.W_EMB(emb_indices)
        noisy_embs = GaussianNoise(0.2)(vec_embs)
        do_embs = Dropout(0.3)(noisy_embs)
        prev_layer = do_embs
        # add one Bi-LSTM layers
        for _ in range(2):
            rnn = Bidirectional(
                LSTM(150, recurrent_dropout=0.25,
                     return_sequences=True,
                     activity_regularizer=l2(L2_COEFF),
                     kernel_regularizer=l2(L2_COEFF),
                     bias_regularizer=l2(L2_COEFF),
                     recurrent_regularizer=l2(L2_COEFF))
            )(prev_layer)
            prev_layer = rnn
        # add simple attention
        attention = Attention(bias=True)(prev_layer)
        # add lexicon-based attention
        lba = LBA(self.lexicon)([lex_indices, prev_layer])
        merged_attention = Average()([attention, lba])
        out = Dense(self._n_y,
                    activation="softmax",
                    activity_regularizer=l2(L2_COEFF),
                    kernel_regularizer=l2(L2_COEFF),
                    bias_regularizer=l2(L2_COEFF))(merged_attention)
        self._model = Model(inputs=[emb_indices, lex_indices], outputs=out)
        self._model.compile(optimizer="adadelta",
                            metrics=["categorical_accuracy"],
                            loss="categorical_hinge")
        self._attention_debug = Model(
            inputs=[emb_indices, lex_indices],
            outputs=[prev_layer, attention]
        )
        self._lba_debug = Model(
            inputs=[emb_indices, lex_indices],
            outputs=[prev_layer, lba]
        )

    def _digitize_data(self, train_x, dev_x):
        """Convert sequences of words to sequences of word and lexicon indices.

        Args:
          train_x (list[list[str]]): training set
          dev_x (list[list[str]]): development set

        Returns:
          2-tuple[list, list]: digitized training and development sets

        """
        # extract word embeddings
        train_embs, dev_embs = super(LBAAnalyzer, self)._digitize_data(
            train_x, dev_x
        )
        train_lex = self._digitize_lex_data(train_x)
        dev_lex = self._digitize_lex_data(dev_x)
        return ([train_embs, train_lex], [dev_embs, dev_lex])

    def _digitize_lex_data(self, dataset):
        """Convert sequences of words to sequences of word and lexicon indices.

        Args:
          dataset (list[tweet]): data set to be digitized

        Returns:
          list[list[int]]: digitized data sets

        """
        # extract word embeddings
        ret = [self._tweet2wseq(tweet_i) for tweet_i in dataset]
        self._wseq2emb_ids(ret, self.get_lex_emb_i)
        ret = self._pad_sequences(ret)
        return ret

    def get_lex_emb_i(self, a_word):
        """Obtain lexicon embedding ind for the given word.

        Args:
          a_word (str):
            word whose embedding index should be retrieved

        Returns:
          int: embedding index of the given word

        """
        a_word = normlex(a_word)
        if a_word in self._w2lex_i:
            return self._w2lex_i[a_word]
        return UNK_IDX

    def _read_lexicons(self, a_lextype2lex, lexicons, encoding=ENCODING):
        """Load lexicons.

        Args:
          a_lextype2lex (dict: lextype -> (dict, dict)): mapping from
            lexicon type to target dictionaries for storing terms (UNUSED)
          lexicons (list): paths to the lexicons to be loaded
          encoding (str): input encoding of the lexicons

        Returns:
          void:

        Note:
          populates `a_pos_term2polscore` and `a_neg_term2polscore` in place

        """
        self._w2lex_i = {EMPTY_TOK: EMPTY_IDX, UNK_TOK: UNK_IDX}
        term2scores = defaultdict(dict)
        lex_pol2score_idx = dict()
        min_score = 1.
        for lexpath_i in lexicons:
            lexname = os.path.splitext(os.path.basename(lexpath_i))[0]
            self._logger.debug("Reading lexicon %s...", lexname)
            lexicon = pd.read_table(lexpath_i, header=None, names=LEX_CLMS,
                                    dtype=LEX_TYPES, encoding=encoding,
                                    error_bad_lines=False, warn_bad_lines=True,
                                    keep_default_na=False, na_values=[''],
                                    quoting=QUOTE_NONE)
            for i, row_i in lexicon.iterrows():
                term = USCORE_RE.sub(' ', row_i.term)
                # since we do not recognize negated context, we skip negated
                # entries
                if NEG_SFX_RE.search(term):
                    self._logger.warn(
                        "Lexicon-based attention does not support negated"
                        " entries.  Skipping term %r.", term)
                    continue
                term = normlex(term)
                lex_pol = (lexname, row_i.polarity)
                if lex_pol not in lex_pol2score_idx:
                    lex_pol2score_idx[lex_pol] = len(lex_pol2score_idx)
                score_idx = lex_pol2score_idx[lex_pol]
                if term not in self._w2lex_i:
                    self._w2lex_i[term] = len(self._w2lex_i)
                word_idx = self._w2lex_i[term]
                score = np.abs(row_i.score)
                term2scores[word_idx][score_idx] = score
                min_score = min(min_score, score)
            self._logger.debug("Lexicon %s read...", lexname)
        # digitize lexicon, converting it to a numpy array
        self.lexicon = np.zeros((len(self._w2lex_i),
                                 len(lex_pol2score_idx)))
        self.lexicon += min_score / 10.
        for w_idx, scores in iteritems(term2scores):
            for score_idx, score in iteritems(scores):
                self.lexicon[w_idx, score_idx] = score
        self.lexicon[EMPTY_IDX, :] = 0.
        return self.lexicon

    def _get_lexicon_w_emb_i(self, a_word):
        """Obtain embedding index for a lexicon term.

        Args:
          a_word (str):
            word whose embedding index should be retrieved

        Returns:
          int:
            embedding index of the given word

        """
        a_word = normlex(a_word)
        if a_word in self._lexicon:
            scores = self._lexicon[a_word]
            max_score = -1.
            best_idx = -1
            for idx, score in iteritems(scores):
                if score > max_score:
                    max_score = score
                    best_idx = idx
            return self.score_idx2emb_idx[best_idx]
        return UNK_IDX
