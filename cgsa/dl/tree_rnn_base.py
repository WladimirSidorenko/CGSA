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

from .base import DLBaseAnalyzer, L2_COEFF
from .layers import EMPTY_IDX, WORD2VEC_LAYER_NAME


##################################################################
# Variables and Constants
DEP_LAYER_NAME = "dependencies"
EMB_INDICES_NAME = "embedding_indices"


##################################################################
# Class
class TreeRNNBaseAnalyzer(DLBaseAnalyzer):
    """Abstract base class for tree-RNNs.

    """

    def predict_proba(self, msg, yvec):
        wseq = self._tweet2wseq(msg)
        deps = self._get_deps(msg)
        self._logger.debug("deps: %r", deps)
        embs = np.array(
            [self.get_test_w_emb(w) for w in wseq]
            + self._pad(len(wseq), self._pad_value), dtype="int32")
        self._logger.debug("embs: %r", embs)
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
        # extract word embeddings
        train_embs, dev_embs = super(TreeRNNBaseAnalyzer, self)._digitize_data(
            train_x, dev_x
        )
        train_deps = pad_sequences([self._get_deps(x) for x in train_x],
                                   value=[-1, -1])
        train_x = [train_deps, train_embs]

        dev_deps = pad_sequences([self._get_deps(x) for x in dev_x],
                                 value=[-1, -1])
        dev_x = [dev_deps, dev_embs]
        return (train_x, dev_x)

    def _get_layer_idx(self, name=WORD2VEC_LAYER_NAME, layers=None):
        """Return index of embedding layer in the model.

        Args:
          name (str): name of the layer whose index should be retrieved
          layers (list or None): list of Keras layers to search in

        Returns:
          int: index of embedding layer

        """
        if layers is None:
            layers = self._model.layers

        for i, layer_i in enumerate(layers):
            if layer_i.name == name:
                return i
        raise KeyError("{:s} layer not found.".format(name))

    def _recompile_model(self, emb_layer_idx):
        """Return the index of embedding layer in the model.

        Args:
          emb_layer_idx (int): index of the embedding layer
          list[keras.Layer]: layers of the new model

        Returns:
          void:

        Note:
          modifies `self._model` in place

        """
        emb_layer = self._model.layers.pop(emb_layer_idx)
        new_emb_layer = Input(shape=emb_layer.output_shape[1:],
                              dtype=emb_layer.dtype,
                              name="embeddings")
        # relink all subsequent layers to receive new input
        old_out, new_out = self._relink_layers(emb_layer, new_emb_layer)
        # replace embedding indices in the input with actual embeddings
        model_inputs = self._model.inputs
        input_idx = self._get_layer_idx(name='/' + EMB_INDICES_NAME,
                                        layers=model_inputs)
        model_inputs[input_idx] = new_emb_layer
        # compare weights
        self._model = self._model.__class__(inputs=model_inputs,
                                            outputs=new_out)
        self._logger.debug(self._model.summary())

    def _relink_layers(self, old_layer, new_layer):
        """Replace .

        Args:
          old_layer (keras.Layer): old layer which should be replaced
          new_layer (Tensor): new layer to be used as replacement

        Returns:
          keras.Layer: newly created output layer

        """
        if len(old_layer.outbound_nodes) == 0:
            # we have reached the final layer, congratulations
            return old_layer, new_layer
        assert len(old_layer.outbound_nodes) == 1, (
            "Multiple layers connected to the old layer (cannot relink)."
            )
        out_node = old_layer.outbound_nodes[0]
        # find next layer
        next_layer = out_node.outbound_layer
        # find out which input of the next layer links to the old layer
        old_output = old_layer.output
        if isinstance(next_layer.input, list):
            new_input = next_layer.input
            j = -1
            for i, inode in enumerate(new_input):
                if inode == old_output:
                    j = i
                    break
            assert j >= 0, (
                "Previous layer not found in the input of the next layer."
            )
            new_input[j] = new_layer
        else:
            new_input = new_layer

        # relink the layer
        def reset_input(layer):
            layer.input_tensors = []
            layer.inbound_nodes = []
            layer.inbound_layers = []
            layer.node_indices = []
            layer.tensor_indices = []
            if hasattr(layer, "_keras_history"):
                del layer._keras_history

        reset_input(next_layer)
        new_next_layer = next_layer(inputs=new_input)
        return self._relink_layers(next_layer, new_next_layer)

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
        self._model.compile(optimizer="adadelta",
                            metrics=["categorical_accuracy"],
                            loss="categorical_hinge")
        self._logger.debug(self._model.summary())

    @abc.abstractmethod
    def _init_rnn(self, *inputs):
        """Abstract method for defining a recurrent unit.

        Returns:
          keras.layer: custom keras layer

        """
        raise NotImplementedError

    def _get_deps(self, tweet):
        """Extract dependency indices of tweet tokens in topological order.

        Args:
          tweet (cgsa.utils.data.Tweet): analyzed input tweet

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
        while active_nodes:
            prnt = active_nodes.pop(0)
            children = deps[prnt]
            active_nodes.extend(children)
            for child_i in children:
                ret.append((child_i, prnt))
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

    def _tweet2wseq(self, msg):
        """Convert tweet to a sequence of word lemmas.

        Args:
          msg (cgsa.data.Tweet): input message

        Return:
          list: lemmas of informative words

        """
        return [w.lemma for w in msg]
