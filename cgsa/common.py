#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Collection of variables and constants common to all modules.

Attributes:

"""

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

import logging

##################################################################
# Variables and Constants
LOG_LVL = logging.DEBUG
LOGGER = logging.getLogger("CGSA")
LOGGER.setLevel(LOG_LVL)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
sh = logging.StreamHandler()
sh.setLevel(LOG_LVL)
sh.setFormatter(formatter)
LOGGER.addHandler(sh)
