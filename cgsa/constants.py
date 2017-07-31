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

# Lexicons
LEX_DIR = os.path.join(DATA_DIR, "lexicons")
GPC_LEX = os.path.join(LEX_DIR, "gpc.txt")
SWS_LEX = os.path.join(LEX_DIR, "sws.txt")
ZRCH_LEX = os.path.join(LEX_DIR, "zrch.txt")
DFLT_MANUAL_LEXICA = [GPC_LEX, SWS_LEX, ZRCH_LEX]

BG_LEX = os.path.join(LEX_DIR,
                      "blair-goldensohn.kim-hovy-seedset.txt")
HU_LIU_LEX = os.path.join(LEX_DIR,
                          "hu-liu.esuli-sebstiani-seedset.txt")
KIM_HOVY_LEX = os.path.join(LEX_DIR,
                            "kim-hovy.turney-littman-seedset.txt")
RR_MINCUT_LEX = os.path.join(LEX_DIR,
                             "rao-ravichandran.min-cut.remus-seedset.txt")
DFLT_AUTO_LEXICA = [BG_LEX, HU_LIU_LEX, KIM_HOVY_LEX]

# Models
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
