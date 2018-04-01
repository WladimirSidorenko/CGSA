#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from csv import QUOTE_NONE
from collections import defaultdict
from keras import backend as K
from keras.models import Model
from keras.layers import (Concatenate, Dense, Dropout, Bidirectional,
                          LSTM, GaussianNoise, Input)
from keras.regularizers import l2
from six import iteritems
import numpy as np
import os
import pandas as pd

from cgsa.utils.common import normlex
from cgsa.base import (LEX_CLMS, LEX_TYPES, NEG_SFX_RE, USCORE_RE)
from cgsa.constants import ENCODING

from .base import (EMB_INDICES_NAME,
                   EMPTY_TOK, UNK_TOK, L2_COEFF)
from .baziotis import BaziotisAnalyzer
from .functional import FunctionalWord2Vec
from .layers import (MergeAttention, RawAttention, CBA, LBA,
                     EMPTY_IDX, UNK_IDX)


##################################################################
# Variables and Constants
LEX_INDICES_NAME = "lexicon_indices"
PRNT_INDICES_NAME = "parent_indices"


##################################################################
# Class
class LBAAnalyzer(FunctionalWord2Vec, BaziotisAnalyzer):
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
        nn_input = self._msg2nn_input(msg)
        ret = self._model.predict(nn_input,
                                  batch_size=1, verbose=2)
        # self._logger.info("ret: %r", ret)
        yvec[:] = ret[0]

    def _init_nn(self):
        self.init_w_emb()
        emb_indices = Input(shape=(self._max_seq_len,),
                            dtype="int32",
                            name=EMB_INDICES_NAME)
        lex_indices = Input(shape=(self._max_seq_len,),
                            dtype="int32",
                            name=LEX_INDICES_NAME)
        prnt_indices = Input(shape=(self._max_seq_len,),
                             dtype="int32",
                             name=PRNT_INDICES_NAME)
        vec_embs = self.W_EMB(emb_indices)
        noisy_embs = GaussianNoise(0.2)(vec_embs)
        do_embs = Dropout(0.3)(noisy_embs)
        prev_layer = do_embs
        # add one Bi-LSTM layers
        for _ in range(1):
            rnn = Bidirectional(
                LSTM(100, recurrent_dropout=0.3,
                     return_sequences=True,
                     activity_regularizer=l2(L2_COEFF),
                     kernel_regularizer=l2(L2_COEFF),
                     bias_regularizer=l2(L2_COEFF),
                     recurrent_regularizer=l2(L2_COEFF))
            )(prev_layer)
            prev_layer = rnn
        # add simple attention
        attention = RawAttention(bias=True)(prev_layer)
        # add lexicon-based attention
        lba = LBA(self.lexicon)([lex_indices, prev_layer])
        # add context-based attention
        cba = CBA()([do_embs, prnt_indices, lba, prev_layer])
        joint_attention = Concatenate(axis=1)([
            MergeAttention()([prev_layer, attention]),
            MergeAttention()([prev_layer, lba]),
            MergeAttention()([prev_layer, cba])
        ])
        out = Dense(self._n_y,
                    activation="softmax",
                    activity_regularizer=l2(L2_COEFF),
                    kernel_regularizer=l2(L2_COEFF),
                    bias_regularizer=l2(L2_COEFF))(joint_attention)
        self._model = Model(inputs=[emb_indices, lex_indices, prnt_indices],
                            outputs=out)
        self._model.compile(**self._train_params)
        self._logger.debug(self._model.summary())

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
        train_deps = self._digitize_dep_data(train_x)
        dev_deps = self._digitize_dep_data(dev_x)
        return ([train_embs, train_lex, train_deps],
                [dev_embs, dev_lex, dev_deps])

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

    def _digitize_dep_data(self, dataset):
        """Convert sequences of words to sequences of word and lexicon indices.

        Args:
          dataset (list[tweet]): data set to be digitized

        Returns:
          list[list[int]]: digitized data sets

        """
        ret = []
        # extract word embeddings
        for tweet_i in dataset:
            offset = self._max_seq_len - len(tweet_i)
            ret.append([w.prnt_idx + offset if w.prnt_idx >= 0 else 0
                        for w in tweet_i])
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
        self.lexicon[UNK_IDX, :] /= 1e2
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

    def _load(self, a_path):
        super(LBAAnalyzer, self)._load(a_path)
        self.attention_debug = K.function(
            inputs=self._model.input + [K.learning_phase()],
            outputs=self._model.get_layer("raw_attention_1").output
        )
        lba_layer = self._model.get_layer("lba_1")
        self._logger.debug("lba_layer: %r", lba_layer.get_weights())
        self.lba_debug = K.function(
            inputs=self._model.input + [K.learning_phase()],
            outputs=self._model.get_layer("lba_1").output
        )
        self.cba_debug = K.function(
            inputs=self._model.input + [K.learning_phase()],
            outputs=self._model.get_layer("cba_1").output
        )

    def reset(self):
        """Remove members which cannot be serialized.

        """
        # set functions to None
        self.attention_debug = None
        self.lba_debug = None
        self.cba_debug = None
        super(LBAAnalyzer, self).reset()

    def debug(self, msg):
        """Output debug information on intermediate steps of prediction.

        """
        nn_input = self._msg2nn_input(msg)
        n = len(msg)
        attention = self.attention_debug(nn_input + [0.])[0][-n:]
        lba = self.lba_debug(nn_input + [0.])[0][-n:]
        cba = self.cba_debug(nn_input + [0.])[0][-n:]
        self._logger.info("msg: %s", msg)
        self._logger.info("attention: %r", attention)
        self._logger.info("lba: %r", lba)
        self._logger.info("cba: %r", cba)
        for i, (tok_i, att_i, lba_i, cba_i) in enumerate(zip(msg, attention,
                                                             lba, cba)):
            self._logger.info(
                "token[%d]: %s; attention: %f; lba: %f; cba: %f;",
                i, tok_i.lemma, att_i, lba_i, cba_i
            )

    def _msg2nn_input(self, msg):
        """Convert input message to embeddings and dependency indices.

        Args:
          msg (cgsa.data.Tweet): input message

        Return:
          (list[array]): input embeddings for the neural network

        """
        assert len(msg) < self._max_seq_len, \
            ("Provided message is longer than the maximum accepted"
             " sequence length: {:d}").format(self._max_seq_len)
        wseq = self._tweet2wseq(msg)
        # obtain word indices
        embs = [np.array(
            self._pad(len(wseq), self._pad_value)
            + [self.get_test_w_emb(w) for w in wseq], dtype="int32")]
        # obtain lexicon indices
        lex_embs = [wseq]
        self._wseq2emb_ids(lex_embs, self.get_lex_emb_i)
        # obtain head indices
        offset = self._max_seq_len - len(msg)
        dep_embs = [self._pad(len(msg))
                    + [w.prnt_idx + offset if w.prnt_idx >= 0 else 0
                       for w in msg]]
        nn_input = [np.asarray(embs), np.asarray(lex_embs),
                    np.asarray(dep_embs)]
        return nn_input
