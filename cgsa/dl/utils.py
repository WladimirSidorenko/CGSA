#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function
from keras import Model
from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib


##################################################################
# Constants
N_GPUS = len([x.name
              for x in device_lib.list_local_devices()
              if x.device_type == 'GPU'])


##################################################################
# Class
class ModelMGPU(Model):
    def __init__(self, ser_model, gpus=N_GPUS):
        """
        Copied from https://github.com/keras-team/keras/issues/2436
        """
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        """Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        """
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)
