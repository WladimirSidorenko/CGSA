#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing abstract base for all sentiment analyzers.

Attributes:
  BaseAnalyzer (class): abstract base for all sentiment analyzers

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals

try:
    from cPickle import dump, load
except ImportError:
    from _pickle import dump, load
import gc
import numpy as np
import os

from cgsa.base import BaseAnalyzer
from cgsa.common import LOGGER
from cgsa.constants import DFLT_MODEL_PATH, MOHAMMAD, SEVERYN, TABOADA
from cgsa.judge import DefaultJudge

##################################################################
# Variables and Constants


##################################################################
# Classes
class SentimentAnalyzer(object):
    """Main class for coarse-grained sentiment analyzers.

    Attributes:
      models (list(BaseSenser)):
        sense disambiguation models
      judge (cgsa.Judge):
        meta-classifier


    """
    @staticmethod
    def load(a_path, on_demand=False):
        """Load serialized model(s) from disc.

        Args:
          a_path (str): path to file from which to load the model
          on_demand (bool): load models later (when explicitly asked for)

        Returns:
          (void)

        """
        # load paths to serialized models
        with open(a_path, "rb") as ifile:
            analyzer = load(ifile)
        analyzer._logger = LOGGER
        # normalize paths to serialized models
        analyzer._dirname = os.path.dirname(a_path)
        if not on_demand:
            analyzer._models = [
                model_i
                for model_i in SentimentAnalyzer._load_models(
                        analyzer
                )
            ]
        return analyzer

    @staticmethod
    def _load_models(a_analyzer):
        """Load serialized sub-model(s) from disc.

        Args:
          a_analyzer (SentimentAnalyzer): instance of this class

        Yields:
          (cgsa.base.BaseAnalyzer): loaded submodel

        """
        for mpath_i in a_analyzer._model_paths:
            with open(os.path.join(a_analyzer._dirname,
                                   mpath_i), "rb") as ifile:
                yield BaseAnalyzer.load(ifile)

    def __init__(self, a_models, *args, **kwargs):
        """Class constructor.

        Args:
          a_models (list[str]): type of the models to train
          a_args (list[str]): arguments to use for initializing models
          a_kwargs (dict): keyword arguments to use for initializing models

        """
        self._models = []
        self.judge = DefaultJudge()

        self._dirname = None
        self._n_cls = 0
        self._cls2idx = {}
        self._idx2cls = {}
        self._wbench = None
        self._logger = LOGGER
        self._model_paths = []
        self._trained_models = []
        self._init_models(a_models, *args, **kwargs)

    def train(self, a_train_data, a_dev_data=None,
              a_path=DFLT_MODEL_PATH, a_grid_search=True,
              a_w2v=False, a_lstsq=False):
        """Train specified model(s) on the provided data.

        Args:
          a_train_data (list or None):
            training set
          a_dev_data (list or None):
            development set
          a_path (str):
            path for storing the model
          a_grid_search (bool):
            use grid search in order to determine hyper-paramaters of
            the model
          a_w2v (bool):
            use word2vec embeddings
          a_lstsq (bool):
            use least squares method

        Returns:
          void:

        """
        # convert polarity classes to integer indices
        self._n_cls = len(
            set(t.label for t in a_train_data) |
            set(t.label for t in (a_dev_data if a_dev_data else []))
        )
        train_x, train_y = self._generate_ts(a_train_data)
        dev_x, dev_y = self._generate_ts(a_dev_data)
        dirname = os.path.dirname(a_path) if a_path else None
        # separately train and dump each model
        for i, model_i in enumerate(self._models):
            try:
                model_i.train(train_x, train_y, dev_x, dev_y,
                              a_grid_search=a_grid_search)
            except:
                # in the end, we are going to switch to the fault-tolerant mode
                raise
            else:
                if dirname:
                    self._save_model(model_i, dirname)
                    self._models[i] = None
        if a_path:
            self._reset()
            with open(a_path, "wb") as ofile:
                dump(self, ofile)

    def batch_predict(self, a_instances):
        """Predict multiple input instances at once.

        Models will be loaded

        Args:
          a_models (list[str]): type of the models to train

        """
        # create a workspace for doing the predictions
        probs = np.zeros((len(a_instances),
                          len(self._model_paths),
                          len(self._cls2idx)
                          ))
        # load each trained model and let it predict the classes
        for i, model_i in enumerate(SentimentAnalyzer._load_models(self)):
            for inst_j, y_j in zip(a_instances, probs):
                model_i.predict_proba(inst_j, y_j[i])
            # unload the moddel to save some disc space
            model_i = None
            gc.collect()
        # let judge merge the decisions
        for inst_j, y_j in zip(a_instances, probs):
            lbl_idx, _ = self.judge.predict(y_j)
            inst_j.label = self._idx2cls[lbl_idx]

    def predict(self, instance):
        """Predict label of a single input instance.

        Args:
          instance (cgsa.data.Tweet): input instance to classify

        Returns:
          str: predicted label

        Note:
          modifies input tweet in place

        """
        if self._wbench is None:
            self._wbench = np.zeros((len(self._models), len(self._cls2idx)))
        else:
            self._wbench *= 0
        # let each trained model predict the probabilities of classes
        for i, model_i in enumerate(self._models):
            model_i.predict_proba(instance, self._wbench[i])
        # let the judge unite the decisions
        lbl_idx, _ = self.judge.predict(self._wbench)
        return self._idx2cls[lbl_idx]

    def save(self, a_path):
        """Dump model to disc.

        Args:
          a_models (list[str]): type of the models to train

        """
        dirname = self._check_path(a_path)
        # store each trained model
        for i, model_i in enumerate(self._models):
            self._save_model(model_i, dirname)
            self._models[i] = model_i = None
            gc.collect()
        self._reset()
        dump(self, a_path)

    def _check_path(self, a_path):
        """Check whether path can be created and is writable.

        Args:
          a_path (str): file path to be checked

        Returns:
          str: directory name of the path

        """
        dirname = os.path.dirname(a_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        elif not os.path.exists(a_path):
            if not os.access(dirname, os.W_OK) or \
               not os.path.isdir(dirname):
                raise RuntimeError("Cannot write to directory '{:s}'.".format(
                    dirname))
        else:
            if not os.access(a_path, os.W_OK):
                raise RuntimeError("Cannot write to file '{:s}'.".format(
                    a_path))
        return dirname

    def _init_models(self, a_models, *a_args, **a_kwargs):
        """Initialize specified models.

        Args:
          a_models (list[str]): type of the models to train
          a_args (list[str]): arguments to use for initializing models
          a_kwargs (dict): keyword arguments to use for initializing models

        Returns:
          void:

        """
        for model_i in a_models:
            if model_i == TABOADA:
                from cgsa.lexicon.taboada import TaboadaAnalyzer
                self._models.append(TaboadaAnalyzer(*a_args, **a_kwargs))
            elif model_i == MOHAMMAD:
                from cgsa.ml.mohammad import MohammadAnalyzer
                self._models.append(MohammadAnalyzer(*a_args, **a_kwargs))
            elif model_i == SEVERYN:
                from cgsa.ml.severyn import SeverynAnalyzer
                self._models.append(SeverynAnalyzer(*a_args, **a_kwargs))

    def _generate_ts(self, a_data):
        """Generate training set.

        Args:
          a_data (list): input instances

        Returns:
          2-tuple(list, list):
            lists of input features and expected classes

        """
        def _check(i, max_i):
            assert i <= max_i, \
                "Number of classes in the data set exceeds maimum number" \
                " of slots."

        x, y = [], []
        if not a_data:
            return (x, y)
        for msg_i in a_data:
            if not msg_i:
                continue
            x.append(msg_i)
            y_i = msg_i.label
            if y_i not in self._cls2idx:
                n = len(self._cls2idx)
                self._idx2cls[n] = y_i
                self._cls2idx[y_i] = n
                _check(len(self._idx2cls), self._n_cls)
            y_i = self._cls2idx[y_i]
            y.append(y_i)
        return (x, y)

    def _prejudge(self, a_rel, a_data):
        """Collect judgments of single classifiers.

        Args:
          a_rel (dict):
            discourse relation whose sense should be predicted
          a_data (2-tuple(dict, dict)):
            list of input JSON data

        Returns:
          np.array: modified ``a_ret``

        """
        if self._wbench is None:
            self._wbench = np.zeros((len(self._models), len(self._cls2idx)))
        else:
            self._wbench *= 0
        for i, model_i in enumerate(self._models):
            model_i.predict(a_rel, a_data, self._wbench, i)
        return self._wbench

    def _reset(self):
        """Remove members which cannot be serialized.

        """
        self._wbench = None
        self._logger = None
        self._models = []

    def _save_model(self, a_model, a_path):
        """Save single model to specifid path.

        Args:
          a_model (cgsa.BaseAnalyzer): prediction model
          a_path (str): file path to be checked

        Returns:
          str: directory name of the path

        """
        relpath = a_model.name + ".model"
        abspath = os.path.join(a_path, relpath)
        self._logger.info("Saving model %s to %s...",
                          a_model.name, abspath
                          )
        with open(abspath, "wb") as ofile:
            a_model.reset()
            dump(a_model, ofile)
        self._logger.info("Model %s saved to %s...",
                          a_model.name, abspath
                          )
        self._model_paths.append(
            os.path.relpath(abspath, a_path)
        )
        return self._model_paths[-1]
