#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Package containing a collection of ML-based sentiment analyzers.

Attributes:

.. moduleauthor:: Wladimir Sidorenko (Uladzimir Sidarenka)

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

from .baziotis import BaziotisAnalyzer
from .severyn import SeverynAnalyzer
from .bilstm import BiLSTMAnalyzer
from .lba import LBAAnalyzer
from .mvrnn import MVRNNAnalyzer
from .rntn import RNTNAnalyzer
from .rnn import RNNAnalyzer
from .yessenalina import YessenalinaAnalyzer

##################################################################
# Variables and Constants
__name__ = "cgsa.dl"
__all__ = ["BaziotisAnalyzer", "BiLSTMAnalyzer", "LBAAnalyzer",
           "MVRNNAnalyzer", "RNNAnalyzer", "RNTNAnalyzer",
           "SeverynAnalyzer", "YessenalinaAnalyzer"]
__author__ = "Uladzimir Sidarenka"
__email__ = "sidarenk at uni dash potsdam dot de"
__version__ = "0.3.0a0"
