#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

"""Module containing custom Keras operations that might be used by different layers.

"""

##################################################################
# Import
from keras import backend as K


##################################################################
# Methods
if K.backend() == "tensorflow":
    def get_subtensor(tensor, *indices):
        indices = K.concatenate(
            [K.expand_dims(idx, 1) for idx in indices], axis=1
        )
        return K.tf.gather_nd(tensor, indices)

    def set_subtensor(tensor, value, *indices):
        # new_tensor = K.zeros_like(tensor)
        # new_tensor += tensor
        indices = K.concatenate(
            [K.expand_dims(idx, 1) for idx in indices], axis=1
        )
        return K.tf.scatter_nd_update(tensor, indices, value)

    def dot_product(x, kernel):
        """
        Wrapper for dot product operation, in order to be compatible with both
        Theano and Tensorflow
        Args:
        x (): input
        kernel (): weights
        Returns:
        """
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)

elif K.backend() == "theano":
    def get_subtensor(tensor, *indices):
        return tensor[indices]

    def set_subtensor(tensor, value, *indices):
        return K.T.set_subtensor(tensor[indices], value)

    def dot_product(x, kernel):
        """
        Wrapper for dot product operation, in order to be compatible with both
        Theano and Tensorflow
        Args:
        x (): input
        kernel (): weights
        Returns:
        """
        return K.dot(x, kernel)

else:
    raise NotImplementedError
