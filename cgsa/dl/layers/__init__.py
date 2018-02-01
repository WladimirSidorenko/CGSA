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

from .word2vec import Word2Vec


##################################################################
# Variables and Constants
EMPTY_IDX = 0
UNK_IDX = 1

__name__ = "cgsa.dl.layers"
__all__ = ["EMPTY_IDX", "UNK_IDX", "Word2Vec"]
__author__ = "Uladzimir Sidarenka"
__email__ = "sidarenk at uni dash potsdam dot de"
__version__ = "0.1.0"
