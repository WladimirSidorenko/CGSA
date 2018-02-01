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

from .severyn import SeverynAnalyzer
from .bilstm import BiLSTMAnalyzer

##################################################################
# Variables and Constants
__name__ = "cgsa.dl"
__all__ = ["BiLSTMAnalyzer", "SeverynAnalyzer"]
__author__ = "Uladzimir Sidarenka"
__email__ = "sidarenk at uni dash potsdam dot de"
__version__ = "0.3.0a0"
