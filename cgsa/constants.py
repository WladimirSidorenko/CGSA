#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Collection of variables and constants common to all modules.

Attributes:
  DFLT_MODEL_PATH (str): path to the main default model file

"""

##################################################################
# Imports
from __future__ import absolute_import

import os


##################################################################
# Variables and Constants
ENCODING = "utf-8"
NFOLDS = 3                      # default number of folds for cross-validation

DIRNAME = os.path.dirname(__file__)
DATA_DIR = os.path.join(DIRNAME, "data")

MODEL_DIR = os.path.join(DATA_DIR, "models")
DFLT_MODEL_PATH = os.path.join(MODEL_DIR, "cgsa.model")

TABOADA = "taboada"
MOHAMMAD = "mohammad"
SEVERYN = "severyn"

# labels
POSITIVE = "positive"
NEGATIVE = "negative"
UNKNOWN = "unknown"
NEUTRAL = "neutral"
MIXED = "mixed"
KNOWN_LABELS = set([POSITIVE, NEGATIVE, UNKNOWN, NEUTRAL, MIXED])
