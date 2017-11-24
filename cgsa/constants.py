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

from csv import QUOTE_NONE
from six import iteritems
from string import punctuation
import os
import pandas as pd
import re


##################################################################
# Variables and Constants
ENCODING = "utf-8"
NFOLDS = 3                      # default number of folds for cross-validation

DIG_RE = re.compile(r"^[\d.]*\d[\d.]*$")
IRR_RE = re.compile(r"[.,:;0-9]+$")
PUNCT_RE = re.compile(r"^(" + '|'.join(
    re.escape(c) for c in punctuation) + ")+$")
SPACE_RE = re.compile(r" +")
USCORE_RE = re.compile(r'_')
SSPACE_RE = re.compile(r"\s\s+")

DIRNAME = os.path.dirname(__file__)
DATA_DIR = os.path.join(DIRNAME, "data")

# Lexicons
LEX_DIR = os.path.join(DATA_DIR, "lexicons")
GPC_LEX = os.path.join(LEX_DIR, "gpc.txt")
SWS_LEX = os.path.join(LEX_DIR, "sws.txt")
ZRCH_LEX = os.path.join(LEX_DIR, "zrch.txt")

BG_LEX = os.path.join(LEX_DIR,
                      "blair-goldensohn.kim-hovy-seedset.txt")
HU_LIU_LEX = os.path.join(LEX_DIR,
                          "hu-liu.esuli-sebstiani-seedset.txt")
KIM_HOVY_LEX = os.path.join(LEX_DIR,
                            "kim-hovy.turney-littman-seedset.txt")
RR_MINCUT_LEX = os.path.join(LEX_DIR,
                             "rao-ravichandran.min-cut.remus-seedset.txt")
TAKAMURA_LEX = os.path.join(LEX_DIR,
                            "takamura.hu-liu-seedset.txt")
KNN_LEX = os.path.join(LEX_DIR,
                       "knn.word2vec.kim_hovy_seedset.txt")
LINPROJ_LEX = os.path.join(LEX_DIR,
                           "linproj.word2vec.kim_hovy_seedset.txt")
# Conditional Probabilities of Lexicon Terms
COND_PROB_FILE = os.path.join(LEX_DIR, "cond-probs")

# Additional Sentiment Resources
INTENSIFIER_PATH = os.path.join(DATA_DIR, "intensifiers.tsv")
INTENSIFIERS = pd.read_table(INTENSIFIER_PATH, header=None,
                             names=("intensifier", "score"),
                             dtype={"intensifier": str,
                                    "score": float},
                             encoding=ENCODING,
                             error_bad_lines=False, warn_bad_lines=True,
                             keep_default_na=False, na_values=[''],
                             quoting=QUOTE_NONE)
# Embeddings
DFLT_W2V_PATH = os.path.join(DATA_DIR, "vectors.word2vec")

# Models
MODEL_DIR = os.path.join(DATA_DIR, "models")
DFLT_MODEL_PATH = os.path.join(MODEL_DIR, "cgsa.model")

BILSTM = "bilstm"
JUREK = "jurek"
KOLCHYNA = "kolchyna"
MOHAMMAD = "mohammad"
MUSTO = "musto"
SEVERYN = "severyn"
TABOADA = "taboada"

# labels
POSITIVE = "positive"
NEGATIVE = "negative"
UNKNOWN = "unknown"
NEUTRAL = "neutral"
MIXED = "mixed"
TWO_CLASS = set([POSITIVE, NEGATIVE])
THREE_CLASS = set([POSITIVE, NEGATIVE, NEUTRAL])
KNOWN_LABELS = set([POSITIVE, NEGATIVE, UNKNOWN, NEUTRAL, MIXED])
CLS2IDX = {NEGATIVE: 0, NEUTRAL: 1, POSITIVE: 2}
IDX2CLS = {v: k for k, v in iteritems(CLS2IDX)}
