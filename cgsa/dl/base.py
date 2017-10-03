#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

try:
    from cPickle import dump, load
except ImportError:
    from _pickle import dump, load
from collections import Counter
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.embeddings import Embedding
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from six import iteritems
from tempfile import mkstemp
import abc
import numpy as np
import os

from cgsa.base import BaseAnalyzer
from cgsa.utils.common import is_relevant, normlex
from cgsa.utils.word2vec import Word2Vec


##################################################################
# Variables and Constants
DFLT_VDIM = 300
EMPTY_IDX = 0
UNK_IDX = 1
DICT_OFFSET = 1
UNK_PROB = 1e-4


##################################################################
# Methods
def tweet2wseq(msg):
    return [normlex(w.lemma)
            for w in msg if is_relevant(w.form)]


##################################################################
# Class
class DLBaseAnalyzer(BaseAnalyzer):
    """Class for DeepLearning-based sentiment analysis.

    Attributes:

    """

    def __init__(self, word2vec=None, lstsq_word2vec=None, **kwargs):
        """Class constructor.

        Args:
          word2vec (str): path to the word2vec file
          lstsq_word2vec (str): path to the word2vec file which should be used
            for least-square computation

        """
        super(DLBaseAnalyzer, self).__init__()
        self.name = "DLBaseAnalyzer"
        if word2vec or lstsq_word2vec:
            self.w2v = Word2Vec  # singleton object
        else:
            self.w2v = None
        self._lstsq = bool(lstsq_word2vec)
        self._plain_w2v = bool(word2vec)
        self.w2emb = None
        self.ndim = -1    # vector dimensionality will be initialized later
        self.intm_dim = -1
        self._model = None
        self._model_path = None
        self._trained = False
        self._n_epochs = 24
        # mapping from word to its embedding index
        self.unk_w_i = 0
        self._aux_keys = set((0, 1))
        self.w_i = 1
        self.w2emb_i = dict()
        self._min_width = 0
        self._n_y = 0

        # variables needed for training
        self._w_stat = self._pred_class = None
        self.W_EMB = self._cost = self._dev_cost = None
        # initialize functions to None
        self._reset_funcs()
        # set up functions for obtaining word embeddings at train and test
        # times
        self._init_wemb_funcs()

    def train(self, train_x, train_y, dev_x, dev_y, a_grid_search):
        self._logger.debug("Training %s...", self.name)
        train_x, train_y, dev_x, dev_y = self._prepare_data(
            train_x, train_y, dev_x, dev_y
        )
        # initialize word embeddings and convert word lists to lists of indices
        self._digitize_data(train_x, dev_x)
        train_x = pad_sequences(train_x)
        dev_x = pad_sequences(dev_x)
        # initialize the network
        self._init_nn()
        # initialize callbacks
        _, ofname = mkstemp(suffix=".hdf5", prefix=self.name + '.')
        try:
            early_stop = EarlyStopping(patience=3, verbose=1)
            chck_point = ModelCheckpoint(monitor="val_acc",
                                         mode="auto", filepath=ofname,
                                         verbose=1, save_best_only=True)
            # start training
            self._model.fit(train_x, train_y,
                            validation_data=(dev_x, dev_y),
                            epochs=self._n_epochs,
                            callbacks=[early_stop, chck_point])
            self._model = load_model(ofname)
            self._trained = True
        finally:
            os.remove(ofname)
        self._logger.debug("%s trained", self.name)

    def predict_proba(self, msg, yvec):
        wseq = tweet2wseq(msg)
        emb_idcs = np.array(
            [self.get_test_w_emb_i(w) for w in wseq]
            + self._pad(len(wseq)), dtype="int32")
        ret = self._model.predict(np.asarray([emb_idcs]),
                                  batch_size=1,
                                  verbose=2)
        for i, prob_i in enumerate(ret[0]):
            yvec[i] = prob_i

    def reset(self):
        """Remove members which cannot be serialized.

        """
        # set functions to None
        self._reset_funcs()
        super(DLBaseAnalyzer, self).reset()

    def save(self, path):
        """Dump model to disc.

        Args:
          a_path (str): file path at which to store the model

        Returns:
          void:

        """
        # set functions to None
        self._model_path = path + ".h5"
        self._model.save(self._model_path)
        self._model = None
        with open(path, "wb") as ofile:
            dump(self, ofile)

    def _load(self):
        super(DLBaseAnalyzer, self)._load()
        self._model = load_model(self._model_path)
        self._init_wemb_funcs()

    @abc.abstractmethod
    def _init_nn(self):
        """Initialize neural network.

        """
        raise NotImplementedError

    def _extract_feats(self, a_tweet):
        pass

    def _init_wemb_funcs(self):
        """Initialize functions for obtaining word embeddings.

        """
        if self._plain_w2v:
            if self.w2v is None:
                self.w2v = Word2Vec
            self.w2v.load()
            self.ndim = self.w2v.ndim
            self.get_train_w_emb_i = self._get_train_w2v_emb_i
            self.init_w_emb = self._init_w2v_emb
            if self._trained:
                self.get_test_w_emb_i = self._get_test_w2v_emb_i
                self._predict_func = self._predict_func_emb
            else:
                self.get_test_w_emb_i = self._get_train_w2v_emb_i
        elif self._lstsq:
            self.ndim = DFLT_VDIM
            self.get_train_w_emb_i = self._get_train_w2v_emb_i
            self.init_w_emb = self._init_w_emb
            if self._trained:
                if self.w2v is None:
                    self.w2v = Word2Vec
                    self.w2v.load()
                self.get_test_w_emb_i = self._get_test_w2v_lstsq_emb_i
                self._predict_func = self._predict_func_emb
            else:
                self.get_test_w_emb_i = self._get_train_w2v_emb_i
        else:
            # checked
            self.ndim = DFLT_VDIM
            self.get_train_w_emb_i = self._get_train_w_emb_i
            self.get_test_w_emb_i = self._get_test_w_emb_i
            self.init_w_emb = self._init_w_emb

    def _reset_funcs(self):
        """Set all compiled theano functions to None.

        Note:
          modifies instance variables in place

        """
        # self._grad_shared = None
        # self._update = None
        # self._predict_class = None
        # self._predict_func = None
        # self._compute_dev_cost = None
        # self._debug_nn = None
        self.get_train_w_emb_i = None
        self.get_test_w_emb_i = None
        self.init_w_emb = None

    def _init_w_emb(self):
        """Initialize task-specific word embeddings.

        """
        self.W_EMB = Embedding(self.w_i, self.ndim,
                               embeddings_initializer="he_normal")

    def _init_w2v_emb(self):
        """Initialize word2vec embedding matrix.

        """
        w_emb = np.empty((self.w_i, self.ndim))
        w_emb[self.unk_w_i, :] = 1e-2  # prevent zeros in this row
        for w, i in iteritems(self.w2emb_i):
            if i == self.unk_w_i:
                continue
            w_emb[i] = self.w2v[w]
        self.W_EMB = theano.shared(value=floatX(w_emb),
                                   name="W_EMB")
        # We unload embeddings every time before the training to free more
        # memory.  Feel free to comment the line below, if you have plenty of
        # RAM.
        self.w2v.unload()

    # def _init_w2emb(self):
    #     """Compute a mapping from Word2Vec to embeddings.

    #     Note:
    #       modifies instance variables in place

    #     """
    #     # construct two matrices - one with the original word2vec
    #     # representations and another one with task-specific embeddings
    #     m = len(self.w2emb_i)
    #     n = self.ndim
    #     j = len(self._aux_keys)
    #     w2v_emb = floatX(np.empty((m, self.w2v.ndim)))
    #     task_emb = floatX(np.empty((m, n)))
    #     k = 0
    #     for w, i in self.w2emb_i.iteritems():
    #         k = i - j
    #         w2v_emb[k] = floatX(self.w2v[w])
    #         task_emb[k] = floatX(self.W_EMB[i])
    #     print("Computing task-specific transform matrix...", end="",
    #           file=sys.stderr)
    #     self.w2emb, res, rank, _ = np.linalg._lstsq(w2v_emb,
    #                                                task_emb)
    #     self.w2emb = floatX(self.w2emb)
    #     print(" done (w2v rank: {:d}, residuals: {:f})".format(rank, sum(res)),
    #           file=sys.stderr)

    def _get_train_w_emb_i(self, a_word):
        """Obtain embedding index for the given word.

        Args:
          a_word (str):
            word whose embedding index should be retrieved

        Returns:
          int:
            embedding index of the given word

        """
        a_word = normlex(a_word)
        if a_word in self.w2emb_i:
            return self.w2emb_i[a_word]
        elif self._w_stat[a_word] < 2 and np.random.binomial(1, UNK_PROB):
            self.w2emb_i[a_word] = self.unk_w_i
            return self.unk_w_i
        else:
            i = self.w2emb_i[a_word] = self.w_i
            self.w_i += 1
            return i

    def _get_test_w_emb_i(self, a_word):
        """Obtain embedding index for the given word.

        Args:
          a_word (str):
            word whose embedding index should be retrieved

        Returns:
          int:
            embedding index od the given word

        """
        a_word = normlex(a_word)
        return self.w2emb_i.get(a_word, self.unk_w_i)

    def _get_train_w2v_emb_i(self, a_word):
        """Obtain embedding index for the given word.

        Args:
          a_word (str):
            word whose embedding index should be retrieved

        Returns:
          int: embedding index of the given word

        """
        a_word = normlex(a_word)
        if a_word in self.w2emb_i:
            return self.w2emb_i[a_word]
        elif a_word in self.w2v:
            i = self.w2emb_i[a_word] = self.w_i
            self.w_i += 1
            return i
        else:
            return self.unk_w_i

    def _get_test_w2v_emb_i(self, a_word):
        """Obtain embedding index for the given word.

        Args:
          a_word (str):
            word whose embedding index should be retrieved

        Returns:
          np.array:
            embedding of the input word

        """
        a_word = normlex(a_word)
        emb_i = self.w2emb_i.get(a_word)
        if emb_i is None:
            if a_word in self.w2v:
                return floatX(self.w2v[a_word])
            return self.W_EMB[self.unk_w_i]
        return self.W_EMB[emb_i]

    def _get_test_w2v_lstsq_emb_i(self, a_word):
        """Obtain embedding index for the given word.

        Args:
          a_word (str):
            word whose embedding index should be retrieved

        Returns:
          np.array:
            embedding of the input word

        """
        a_word = normlex(a_word)
        emb_i = self.w2emb_i.get(a_word)
        if emb_i is None:
            if a_word in self.w2v:
                return floatX(np.dot(self.w2v[a_word], self.w2emb))
            return self.W_EMB[self.unk_w_i]
        return self.W_EMB[emb_i]

    def _prepare_data(self, train_x, train_y, dev_x, dev_y):
        """Provide train/test split and digitize the data.

        """
        if not dev_x:
            n = len(train_x)
            n_dev = int(n / 15)
            idcs = list(range(n))
            np.random.shuffle(idcs)

            def get_split(data, idcs):
                return [data[i] for i in idcs]

            dev_x = get_split(train_x, idcs[:n_dev])
            dev_y = get_split(train_y, idcs[:n_dev])
            train_x = get_split(train_x, idcs[n_dev:])
            train_y = get_split(train_y, idcs[n_dev:])

        # convert tweets to word lists
        train_x = [tweet2wseq(x) for x in train_x]
        dev_x = [tweet2wseq(x) for x in dev_x]
        self._n_y = len(set(train_y + dev_y))
        train_y = to_categorical(np.asarray(train_y))
        dev_y = to_categorical(np.asarray(dev_y))
        return (train_x, train_y, dev_x, dev_y)

    def _compute_w_stat(self, train_x):
        """Compute word frequencies on the corpus.

        Args:
          train_x (list[list[str]]): training instances

        Returns:
          void:

        Note:
          modifies instance variables in place

        """
        self._w_stat = Counter(w for t in train_x for w in t)

    def _digitize_data(self, train_x, dev_x):
        """Convert sequences of words to sequences of word indices.

        Args:
          train_x (list[list[str]]): training set
          dev_x (list[list[str]]): development set

        Returns:
          void:

        Note:
          modifies input arguments in-place

        """
        self._compute_w_stat(train_x)

        def wseq2emb_ids(data, w2i):
            for i, inst_i in enumerate(data):
                data[i] = np.asarray([w2i(w) for w in inst_i]
                                     + self._pad(len(inst_i)),
                                     dtype="int32")

        wseq2emb_ids(train_x, self.get_train_w_emb_i)
        wseq2emb_ids(dev_x, self.get_test_w_emb_i)

    def _pad(self, xlen):
        """Add indices of empty words to match minimum filter length.

        Args:
          xlen (int): length of the input instance

        """
        return [EMPTY_IDX] * max(0, self._min_width - xlen)
