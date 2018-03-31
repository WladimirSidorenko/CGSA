#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

"""Abstract base class for handling word2vec embeddings in functional models.

"""

##################################################################
# Imports
from __future__ import (absolute_import, unicode_literals, print_function,
                        print_function)
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
        self._logger.info("emb_layer: %r", emb_layer)
        new_emb_layer = Input(shape=emb_layer.output_shape[1:],
                              dtype=emb_layer.dtype,
                              name="embeddings")
        self._logger.info("new_emb_layer: %r", new_emb_layer)
        # relink all subsequent layers to receive new input
        input_layers = self._model.input_layers
        self._logger.info("self._model.input_layers: %r", input_layers)
        old2new = {layer.name: None for layer in input_layers}
        old2new[emb_layer.name] = new_emb_layer
        self._logger.info("*** old2new: %r (before relinking)", old2new)
        self._relink_layers(input_layers, old2new)
        self._logger.info("*** old2new: %r (after relinking)", old2new)
        # replace embedding indices in the input with the newly created
        # embedding layer
        model_inputs = self._model.inputs
        self._logger.info("self._model.inputs (original): %r", model_inputs)
        input_idx = self._get_layer_idx(name='/' + EMB_INDICES_NAME,
                                        layers=model_inputs)
        model_inputs[input_idx] = new_emb_layer
        self._logger.info("self._model.inputs (modified): %r", model_inputs)
        # replace model's output with the newly relinked layers
        model_outputs = self._model.output_layers
        self._logger.info("self._model.outputs: %r", model_outputs)
        new_outputs = [
            layer
            if old2new[layer.name] is None
            else old2new[layer.name]
            for layer in model_outputs
        ]
        self._model = self._model.__class__(inputs=model_inputs,
                                            outputs=new_outputs)
        self._logger.debug(self._model.summary())

    def _relink_layers(self, layers, old2new, start=False):
        """Replace input layer with pre-trained embedding and propagate the changes.

        Args:
          layers (list[keras.Layer]): old layers that possibly should be
            replaced
          old2new (dict[keras.Layer]): mapping from old layers to the new ones
          start (bool): whether we are at the beginning of the relinking
            process

        Returns:
          void:

        Note:
          modifies `old2new` in place

        """
        if not layers:
            # we have visited all layers of the network, congratulations
            return

        # reset layer's memory
        def reset_input(layer):
            layer.input_tensors = []
            layer._inbound_nodes = []
            layer.inbound_layers = []
            layer.node_indices = []
            layer.tensor_indices = []
            # layer._outbound_nodes = []
            if hasattr(layer, "_keras_history"):
                del layer._keras_history

        old_layer = layers.pop(0)
        out_nodes = [node_i for node_i in old_layer._outbound_nodes]
        for out_node in out_nodes:
            changed = False
            # find next layer
            next_layer = out_node.outbound_layer
            next_name = next_layer.name
            if next_name in old2new:
                layers.append(next_layer)
                continue
            # find out which inputs of the next layer need to be replaced
            next_input = out_node.inbound_layers
            if len(next_input) > 1:
                new_input = []
                for layer_i in next_input:
                    layer_name = layer_i.name
                    if layer_name not in old2new:
                        break
                    elif old2new[layer_name] is None:
                        new_input.append(layer_i.output)
                    else:
                        changed = True
                        new_input.append(old2new[layer_name])
                # if haven't seen all inputs yet, then do nothing for the time
                # being, since we will eventually discover this layer later
                # again while processing other inputs
                if len(new_input) != len(next_input):
                    continue
                next_input = [layer.output for layer in next_input]
            else:
                next_input = next_input[0]
                next_input_name = next_input.name
                assert next_input_name in old2new, \
                    "Missing input to the layer {!s}: {!r}.".format(
                        next_layer, next_input_name)
                if old2new[next_input_name] is None:
                    new_input = next_input
                else:
                    changed = True
                    new_input = old2new[next_input_name]
            layers.append(next_layer)
            # if the input to the layer hasn't changed, then we need not to
            # change the layer either
            if changed:
                reset_input(next_layer)
                new_next_layer = next_layer(inputs=new_input)
                old2new[next_name] = new_next_layer
            # otherwise, introduce a new layer and remember it
            else:
                old2new[next_layer.name] = None
        del old_layer._outbound_nodes[:]
        return self._relink_layers(layers, old2new)
