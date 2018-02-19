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
from .mvrn import MVRN
from .msrn import MSRN
from .rn import RN
from .rnt import RNT
from .membeddings import MatrixEmbedding
from .ymembeddings import YMatrixEmbedding
from .word2vec import WORD2VEC_LAYER_NAME, Word2Vec


##################################################################
# Variables and Constants
EMPTY_IDX = 0
UNK_IDX = 1
CUSTOM_OBJECTS = {"Attention": Attention, "MatrixEmbedding": MatrixEmbedding,
                  "Word2Vec": Word2Vec, "MVRN": MVRN, "MSRN": MSRN, "RN": RN,
                  "RNT": RNT, "YMatrixEmbedding": YMatrixEmbedding}

__name__ = "cgsa.dl.layers"
__all__ = ["Attention", "EMPTY_IDX", "UNK_IDX", "MVRN", "RN", "RNT",
           "MatrixEmbedding", "WORD2VEC_LAYER_NAME", "Word2Vec",
           "CUSTOM_OBJECTS"]
__author__ = "Uladzimir Sidarenka"
__email__ = "sidarenk at uni dash potsdam dot de"
__version__ = "0.1.0"
