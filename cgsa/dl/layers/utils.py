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
    def dot_product(x, kernel):
        """
        Wrapper for dot product operation, in order to be compatible with both
        Theano and Tensorflow
        Args:
        x (): input
        kernel (): weights
        Returns:
        """
        return K.squeeze(K.dot(x,
                               K.expand_dims(K.tf.transpose(kernel))),
                         axis=-1)

    def get_subtensor(tensor, *indices):
        indices = K.concatenate(
            [K.expand_dims(idx, 1) for idx in indices], axis=1
        )
        return K.tf.gather_nd(tensor, indices)

    def set_subtensor(tensor, value, *indices):
        x_shape = K.shape(indices[-1])[0]
        z_shape = K.int_shape(tensor)[-1]
        indices = [repeat_elements(idx, z_shape) for idx in indices]
        indices = K.tf.transpose(K.tf.stack(indices))
        z = K.expand_dims(
            K.tf.reshape(
                K.tf.tile(K.arange(0, z_shape), [x_shape]), [-1]), -1
        )
        indices = K.concatenate([indices, z], axis=-1)
        binary_mask = K.tf.cast(
            K.tf.sparse_to_dense(indices, K.shape(tensor), 1),
            K.tf.bool)
        val_tensor = K.tf.sparse_to_dense(indices, K.shape(tensor),
                                          K.flatten(value))
        return K.tf.where(binary_mask, val_tensor, tensor)

    def repeat_elements(tensor, n, axis=0):
        return K.tf.reshape(
            K.tf.tile(
                K.tf.expand_dims(tensor, -1), [1, n]), [-1])

elif K.backend() == "theano":
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

    def get_subtensor(tensor, *indices):
        return tensor[indices]

    def set_subtensor(tensor, value, *indices):
        return K.T.set_subtensor(tensor[indices], value)

    def repeat_elements(tensor, n, axis=0):
        return K.repeat_elements(tensor, n, axis)


else:
    raise NotImplementedError
