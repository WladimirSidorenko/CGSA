#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Package containing a collection of custom Keras layers.

Attributes:

.. moduleauthor:: Wladimir Sidorenko (Uladzimir Sidarenka)

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from .attention import Attention
from .word2vec import Word2Vec


##################################################################
# Variables and Constants
EMPTY_IDX = 0
UNK_IDX = 1
CUSTOM_OBJECTS = {"Attention": Attention, "Word2Vec": Word2Vec}

__name__ = "cgsa.dl.layers"
__all__ = ["Attention", "EMPTY_IDX", "UNK_IDX", "Word2Vec",
           "CUSTOM_OBJECTS"]
__author__ = "Uladzimir Sidarenka"
__email__ = "sidarenk at uni dash potsdam dot de"
__version__ = "0.1.0"
