#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

"""Abstract base class for handling word2vec embeddings in functional models.

"""

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function, print_function
from abc import ABC
from keras.layers import Input

from .base import EMB_INDICES_NAME
from .layers import WORD2VEC_LAYER_NAME


##################################################################
# Class
class FunctionalWord2Vec(ABC):
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
        """Replace the embedding layer of the model.

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
        """Replace input layer with embeddings.

        Args:
          old_layer (keras.Layer): old layer which should be replaced
          new_layer (Tensor): new layer to be used as replacement

        Returns:
          keras.Layer: newly created output layer

        """
        if len(old_layer._outbound_nodes) == 0:
            # we have reached the final layer, congratulations
            return old_layer, new_layer
        assert len(old_layer._outbound_nodes) == 1, (
            "Multiple layers connected to the old layer (cannot relink)."
            )
        out_node = old_layer._outbound_nodes[0]
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
            layer._inbound_nodes = []
            layer.inbound_layers = []
            layer.node_indices = []
            layer.tensor_indices = []
            if hasattr(layer, "_keras_history"):
                del layer._keras_history

        reset_input(next_layer)
        new_next_layer = next_layer(inputs=new_input)
        return self._relink_layers(next_layer, new_next_layer)
