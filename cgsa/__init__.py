#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Main meta-package containing a collection of sentiment analyzers.

Attributes:
  lexicon (module):
    routines common to multiple subpackages
  dl (subpackage):
    rule-based discourse segmenter for Mate dependency trees
  ml (subpackage):
    auxiliary segmenter routines used by syntax-driven segmenters
  evaluation (subpackage):
    metrics for evaluating discourse segmentation

.. moduleauthor:: Wladimir Sidorenko (Uladzimir Sidarenka)

"""

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function


##################################################################
# Variables and Constants
__name__ = "cgsa"
__all__ = ["lexicon", "dl", "ml", "evaluation"]
__author__ = "Uladzimir Sidarenka"
__email__ = "sidarenk at uni dash potsdam dot de"
__version__ = "0.1.0a0"
