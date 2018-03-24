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
from keras.regularizers import l2
from keras.utils import to_categorical
from six import iteritems
from tempfile import mkstemp
import abc
import numpy as np
import os

from cgsa.base import BaseAnalyzer
from cgsa.utils.common import LOGGER, is_relevant, normlex
from .layers import CUSTOM_OBJECTS, EMPTY_IDX, UNK_IDX
from .layers.word2vec import Word2Vec


##################################################################
# Variables and Constants
# default dimensionality for task-specific vectors
DFLT_VDIM = 100
DFLT_N_EPOCHS = 24  # 24

EMPTY_TOK = "%EMPTY%"
UNK_TOK = "%UNK%"
DICT_OFFSET = 1
UNK_PROB = 1e-4
L2_COEFF = 1e-4
EMB_INDICES_NAME = "embedding_indices"


##################################################################
# Methods


##################################################################
# Class
class DLBaseAnalyzer(BaseAnalyzer):
    """Class for DeepLearning-based sentiment analysis.

    Attributes:

    """

    def __init__(self, w2v=False, lstsq=False, embeddings=None, **kwargs):
        """Class constructor.

        Args:
          w2v (bool): use word2vec embeddings
          lstsq (bool): use the least squares method
          embeddings (cgsa.utils.word2vec.Word2Vec or None): pretrained
            embeddings

        """
        super(DLBaseAnalyzer, self).__init__()
        self.name = "DLBaseAnalyzer"
        # boolean flags indicating whether to use external embeddings
        self._w2v = w2v
        self._lstsq = lstsq
        # actual external embeddings
        self._embeddings = embeddings
        # mapping from words to their embedding indices in `self._embs` or
        # `self.W_EMB`
        self._w2i = {EMPTY_TOK: EMPTY_IDX, UNK_TOK: UNK_IDX}
        self._pad_value = EMPTY_IDX
        # mapping from words to their embeddings (will be initialized after
        # training the network, if `w2v` or `lstsq` are true)
        self._embs = None
        # least squares matrix (will be initialized after training the network,
        # if true)
        self._lstsq_mtx = None
        self.ndim = -1    # vector dimensionality will be initialized later
        self.intm_dim = -1
        self._model = None
        self._model_path = None
        self._trained = False
        self._n_epochs = DFLT_N_EPOCHS
        # mapping from word to its embedding index
        self._aux_keys = set((0, 1))
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
        self._start_training()
        self._logger.debug("Training %s...", self.name)
        self._logger.debug("Preparing dataset...")
        train_x, train_y, dev_x, dev_y = self._prepare_data(
            train_x, train_y, dev_x, dev_y
        )
        self._logger.debug("Dataset ready...")
        # initialize the network
        self._logger.debug("Initializing the network...")
        self._init_nn()
        self._logger.debug("Network ready...")
        # initialize callbacks
        _, ofname = mkstemp(suffix=".hdf5", prefix=self.name + '.')
        try:
            early_stop = EarlyStopping(patience=3, verbose=1)
            chck_point = ModelCheckpoint(filepath=ofname,
                                         monitor="val_categorical_accuracy",
                                         mode="auto",
                                         verbose=1,
                                         save_best_only=True)
            self._model.fit(train_x, train_y,
                            validation_data=(dev_x, dev_y),
                            epochs=self._n_epochs,
                            callbacks=[early_stop, chck_point])
            self._model = load_model(ofname, custom_objects=CUSTOM_OBJECTS)
            self._finish_training()
        finally:
            os.remove(ofname)
        self._logger.debug("%s trained", self.name)

    def predict_proba(self, msg, yvec):
        wseq = self._tweet2wseq(msg)
        embs = np.array(
            [self._pad(len(wseq), self._pad_value)
             + self.get_test_w_emb(w) for w in wseq], dtype="int32")
        ret = self._model.predict(np.asarray([embs]),
                                  batch_size=1,
                                  verbose=2)
        yvec[:] = ret[0]

    def predict_proba_raw(self, messages):
        yvecs = np.zeros((len(messages), self._n_y))
        for i, msg_i in enumerate(messages):
            self.predict_proba(msg_i, yvecs[i])
        return yvecs

    def restore(self, embs):
        """Restore members which could not be serialized.

        Args:
          embs (cgsa.utils.word2vec.Word2Vec or None): pretrained
            embeddings

        """
        self._embeddings = embs
        self._logger = LOGGER
        self._init_wemb_funcs()

    def reset(self):
        """Remove members which cannot be serialized.

        """
        # set functions to None
        self._reset_funcs()
        self._embeddings = None
        self.W_EMB = None
        super(DLBaseAnalyzer, self).reset()

    def save(self, path):
        """Dump model to disc.

        Args:
          a_path (str): file path at which to store the model

        Returns:
          void:

        """
        # set functions to None
        model_path = path + ".h5"
        self._model.save(model_path)
        self._model_path = os.path.basename(model_path)
        # all paths are relative
        model = self._model
        self._model = None
        with open(path, "wb") as ofile:
            dump(self, ofile)
        self._model = model

    def _load(self, a_path):
        super(DLBaseAnalyzer, self)._load(a_path)
        self._model = load_model(
            os.path.join(a_path, self._model_path),
            custom_objects=CUSTOM_OBJECTS
        )

    @abc.abstractmethod
    def _init_nn(self):
        """Initialize neural network.

        """
        raise NotImplementedError

    def _extract_feats(self, a_tweet):
        pass

    def _start_training(self):
        """Prepare for training.

        """
        self._trained = False

    def _finish_training(self):
        """Finalize the trained network.

        """
        self._logger.info("Finalizing network")
        if self._lstsq or self._w2v:
            emb_layer_idx = self._get_layer_idx()
            if self._lstsq:
                # Extract embeddings from the network
                task_embs = self._model.layers[emb_layer_idx].get_weights()
                assert len(task_embs) == 1, \
                    ("Unmatching number of trained paramaters:"
                     " {:d} instead of {:d}").format(
                         len(task_embs), 1)
                task_embs = task_embs[0]
                # extract only embeddings of known words
                START_IDX = UNK_IDX + 1
                w2v_embs = self._embs
                # Compute the least square matrix
                self._logger.info("Computing transform matrix for"
                                  " task-specific embeddings.")
                self._lstsq_mtx, res, rank, _ = np.linalg.lstsq(
                    w2v_embs[START_IDX:], task_embs[START_IDX:]
                )
                self._logger.info("Transform matrix computed"
                                  " (rank: %d, residuals: %f).",
                                  rank, sum(res))
                self._embs = task_embs
            # pop embedding layer and modify the first layer coming after it to
            # accept plaing embeddings as input
            self._recompile_model(emb_layer_idx)
            self._pad_value = self._embs[EMPTY_IDX]
        self._logger.info("Network finalized")
        self._trained = True

    def _get_layer_idx(self):
        """Return the index of embedding layer in the model.

        Args:
          name (str): name of the layer (IGNORED)

        Returns:
          int: index of embedding layer

        """
        return 0

    def _recompile_model(self, emb_layer_idx):
        """Change model by removing the embedding layer and .

        Args:
          emb_layer_idx (int): index of the embedding layer

        Returns:
          void:

        Note:
          modifies `self._model` in place

        """
        layers = self._model.layers
        emb_layer = layers.pop(emb_layer_idx)
        first_layer = layers.pop(emb_layer_idx)
        layer_config = first_layer.get_config()
        layer_config["input_shape"] = (None, emb_layer.output_dim)
        new_layer = first_layer.__class__.from_config(
            layer_config
        )
        new_layer.build((emb_layer.input_dim, emb_layer.output_dim))
        new_layer.set_weights(first_layer.get_weights())
        layers.insert(emb_layer_idx, new_layer)
        self._model = self._model.__class__(layers=layers)

    def _init_wemb_funcs(self):
        """Initialize functions for obtaining word embeddings.

        """
        if self.ndim < 0:
            self.ndim = DFLT_VDIM
        if self._w2v:
            self._embeddings.load()
            self.ndim = self._embeddings.ndim
            self.init_w_emb = self._init_w2v_emb
            self.get_train_w_emb_i = self._get_train_w2v_emb_i
            if self._trained:
                self.get_test_w_emb = self._get_test_w2v_emb
            else:
                self.get_test_w_emb = self._get_train_w2v_emb_i
        elif self._lstsq:
            self._embeddings.load()
            self.ndim = self._embeddings.ndim
            self.init_w_emb = self._init_w2v_emb
            self.get_train_w_emb_i = self._get_train_w2v_emb_i
            if self._trained:
                self.get_test_w_emb = self._get_test_w2v_lstsq_emb
            else:
                self.get_test_w_emb = self._get_train_w2v_emb_i
        else:
            # checked
            self.init_w_emb = self._init_w_emb
            self.get_train_w_emb_i = self._get_train_w_emb_i
            self.get_test_w_emb = self._get_test_w_emb_i

    def _reset_funcs(self):
        """Set all compiled theano functions to None.

        Note:
          modifies instance variables in place

        """
        self.get_train_w_emb_i = None
        self.get_test_w_emb_i = None
        self.init_w_emb = None

    def _init_w_emb(self):
        """Initialize task-specific word embeddings.

        """
        self.W_EMB = Embedding(len(self._w2i), self.ndim,
                               embeddings_initializer="he_normal",
                               embeddings_regularizer=l2(L2_COEFF))

    def _init_w2v_emb(self):
        """Initialize word2vec embedding matrix.

        """
        self._embeddings.load()
        self.ndim = self._embeddings.ndim
        self._embs = np.empty((len(self._w2i), self.ndim))
        self._embs[EMPTY_IDX, :] *= 0
        self._embs[UNK_IDX, :] = 1e-2  # prevent zeros in this row
        for w, i in iteritems(self._w2i):
            if i == EMPTY_IDX or i == UNK_IDX:
                continue
            self._embs[i] = self._embeddings[w]
        # initialize custom keras layer
        self.W_EMB = Word2Vec(self._embs, trainable=self._lstsq)
        # We unload embeddings every time before the training to free more
        # memory.  Feel free to comment the line below, if you have plenty of
        # RAM.
        self._embeddings.unload()

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
        if a_word in self._w2i:
            return self._w2i[a_word]
        elif self._w_stat[a_word] < 2 and np.random.binomial(1, UNK_PROB):
            return UNK_IDX
        else:
            i = self._w2i[a_word] = len(self._w2i)
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
        return self._w2i.get(a_word, UNK_IDX)

    def _get_train_w2v_emb_i(self, a_word):
        """Obtain embedding index for the given word.

        Args:
          a_word (str):
            word whose embedding index should be retrieved

        Returns:
          int: embedding index of the given word

        """
        a_word = normlex(a_word)
        if a_word in self._w2i:
            return self._w2i[a_word]
        elif a_word in self._embeddings:
            i = self._w2i[a_word] = len(self._w2i)
            return i
        else:
            return UNK_IDX

    def _get_test_w2v_emb(self, a_word):
        """Obtain embedding index for the given word.

        Args:
          a_word (str):
            word whose embedding index should be retrieved

        Returns:
          np.array:
            embedding of the input word

        """
        a_word = normlex(a_word)
        emb_i = self._w2i.get(a_word)
        if emb_i is None:
            if a_word in self._embeddings:
                return self._embeddings[a_word]
            return self._embs[UNK_IDX]
        return self._embs[emb_i]

    def _get_test_w2v_lstsq_emb(self, a_word):
        """Obtain embedding index for the given word.

        Args:
          a_word (str):
            word whose embedding index should be retrieved

        Returns:
          np.array:
            embedding of the input word

        """
        a_word = normlex(a_word)
        emb_i = self._w2i.get(a_word)
        if emb_i is None:
            if a_word in self._embeddings:
                return np.dot(self._embeddings[a_word],
                              self._lstsq_mtx)
            return self._embs[UNK_IDX]
        return self._embs[emb_i]

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

        # convert tweets to word indices
        train_x, dev_x = self._digitize_data(train_x, dev_x)
        self._n_y = len(set(train_y) | set(dev_y))
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
          2-tuple[list, list]: digitized training and development sets

        """
        train_x = [self._tweet2wseq(x) for x in train_x]
        dev_x = [self._tweet2wseq(x) for x in dev_x]
        self._compute_w_stat(train_x)

        self._wseq2emb_ids(train_x, self.get_train_w_emb_i)
        self._wseq2emb_ids(dev_x, self.get_test_w_emb)
        train_x = self._pad_sequences(train_x)
        dev_x = self._pad_sequences(dev_x)
        return (train_x, dev_x)

    def _pad(self, xlen, pad_value=EMPTY_IDX):
        """Add indices or vectors of empty words to match minimum filter length.

        Args:
          xlen (int): length of the input instance

        """
        return [pad_value] * max(0, self._min_width - xlen)

    def _pad_sequences(self, x):
        """Make all input instances of equal length.

        Args:
          x (list[np.array]): list of embedding indices

        Returns:
          x: list of embedding indices of equal lengths

        """
        return pad_sequences(x)

    def _tweet2wseq(self, msg):
        """Convert tweet to a sequence of word lemmas if these words are informative.

        Args:
          msg (cgsa.data.Tweet): input message

        Return:
          list: lemmas of informative words

        """
        return [normlex(w.lemma)
                for w in msg if is_relevant(w.form)]

    def _wseq2emb_ids(self, data, w2i):
        """Convert sequence of words to embedding indices.

        Args:
          data (list[str]): list of input words
          w2i (func): function to convert words to embedding indices

        Return:
          list[int]: list of embedding indices

        """
        for i, inst_i in enumerate(data):
            data[i] = np.asarray(
                self._pad(len(inst_i))
                + [w2i(w) for w in inst_i], dtype="int32")
