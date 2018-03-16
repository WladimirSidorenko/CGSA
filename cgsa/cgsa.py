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
from six import iteritems
try:
    from cPickle import dump, load
except ImportError:
    from _pickle import dump, load

import gc
import numpy as np
import os

from cgsa.base import BaseAnalyzer
from cgsa.utils.common import LOGGER
from cgsa.constants import (DFLT_MODEL_PATH, DFLT_W2V_PATH, BAZIOTIS,
                            BILSTM, GAMON, GUENTHER, HU_LIU, JUREK,
                            KOLCHYNA, LBA, MOHAMMAD, MUSTO, MVRNN, RNN, RNTN,
                            SEVERYN, TABOADA, YESSENALINA, CLS2IDX, IDX2CLS)
from cgsa.dl.base import DLBaseAnalyzer
from cgsa.judge import DefaultJudge
from cgsa.utils.word2vec import Word2Vec


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
    def load(a_path, w2v_path, on_demand=False):
        """Load serialized model(s) from disc.

        Args:
          a_path (str): path to file from which to load the model
          on_demand (bool): load models later (when explicitly asked for)

        Returns:
          (void)

        """
        # load paths to serialized models
        LOGGER.debug("Loading analyzer from file: %s", a_path)
        with open(a_path, "rb") as ifile:
            analyzer = load(ifile)
        # normalize paths to serialized models
        analyzer._dirname = os.path.dirname(a_path)
        analyzer._logger = LOGGER
        if analyzer._w2v or analyzer._lstsq:
            if w2v_path != analyzer._w2v_path:
                LOGGER.warn(
                    "Classifier %r was trained with a different embedding"
                    " file: trained with %s vs. testing with %s", analyzer,
                    analyzer._w2v_path, w2v_path
                )
            analyzer._embeddings = Word2Vec(w2v_path)
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
            a_analyzer._logger.debug(
                "Loading model from file: %s (dirname: %s)",
                mpath_i, a_analyzer._dirname)
            with open(os.path.join(a_analyzer._dirname,
                                   mpath_i), "rb") as ifile:
                model_i = BaseAnalyzer.load(a_analyzer._dirname, ifile)
                model_i.restore(a_analyzer._embeddings)
                yield model_i

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
        self._wbench = None
        self._logger = LOGGER
        self._model_paths = []
        self._trained_models = []
        self._w2v = kwargs.get("w2v", False)
        self._lstsq = kwargs.get("lstsq", False)
        self._w2v_path = os.path.abspath(
            kwargs.pop("w2v_path", DFLT_W2V_PATH)
        )
        self._embeddings = None
        if self._w2v or self._lstsq:
            self._embeddings = Word2Vec(self._w2v_path)
        kwargs["embeddings"] = self._embeddings
        self._init_models(a_models, *args, **kwargs)

    def train(self, a_train_data, a_dev_data=None,
              a_path=DFLT_MODEL_PATH, a_grid_search=True):
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
            LOGGER.debug("Saving analyzer in %s...", a_path)
            self._reset()
            with open(a_path, "wb") as ofile:
                dump(self, ofile)
            LOGGER.debug("Analyzer saved...")

    def batch_predict(self, a_instances):
        """Predict multiple input instances at once.

        Models will be loaded

        Args:
          a_models (list[str]): type of the models to train

        """
        # create a workspace for doing the predictions
        probs = np.zeros((len(a_instances),
                          len(self._model_paths),
                          len(IDX2CLS)
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
            inst_j.label = IDX2CLS[lbl_idx]

    def predict(self, instance):
        """Predict label of a single input instance.

        Args:
          instance (cgsa.utils.data.Tweet): input instance to classify

        Returns:
          str: predicted label

        Note:
          modifies input tweet in place

        """
        if self._wbench is None:
            self._wbench = np.zeros((len(self._models), len(CLS2IDX)))
        else:
            self._wbench *= 0
        # let each trained model predict the probabilities of classes
        for i, model_i in enumerate(self._models):
            model_i.predict_proba(instance, self._wbench[i])
        # let the judge unite the decisions
        lbl_idx, _ = self.judge.predict(self._wbench)
        return IDX2CLS[lbl_idx]

    def debug(self, instance):
        """Explain predictions of each classifier.

        Args:
          instance (cgsa.utils.data.Tweet): input instance to classify

        Returns:
          str: predicted label

        Note:
          modifies input tweet in place

        """
        n_classes = len(CLS2IDX)
        if self._wbench is None:
            self._wbench = np.zeros((len(self._models), n_classes))
        else:
            self._wbench *= 0

        from .explainer import Explainer
        explainer = Explainer(class_names=IDX2CLS)
        # let each trained model predict the probabilities of classes
        for i, model_i in enumerate(self._models):
            self._logger.info("Considering model (%d): %r", i, model_i)
            model_i.predict_proba(instance, self._wbench[i])
            self._logger.info("Predicted scores: %r", self._wbench[i])
            # explanations
            explanations = explainer.explain_instance(
                instance,
                model_i.predict_proba_raw,
                num_features=6, labels=[y for y in range(len(IDX2CLS))]
            )
            for i, cls in iteritems(IDX2CLS):
                self._logger.info("Class: %s", cls)
                explanation = explanations.as_list(label=i)
                self._logger.info("Explanation: %r", explanation)
        # let the judge unite the decisions
        self._logger.info("All predicted scores: %r", self._wbench)
        lbl_idx, _ = self.judge.predict(self._wbench)
        self._logger.info("Judge prediction: %r", self._wbench)
        return IDX2CLS[lbl_idx]

    def save(self, a_path):
        """Dump model to disc.

        Args:
          a_models (list[str]): type of the models to train

        """
        dirname = self._check_path(a_path)
        # store each trained model
        for i, model_i in enumerate(self._models):
            self._logger.debug("Saving model in %s", dirname)
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
            if model_i == BAZIOTIS:
                from cgsa.dl.baziotis import BaziotisAnalyzer
                analyzer_cls = BaziotisAnalyzer
            elif model_i == BILSTM:
                from cgsa.dl.bilstm import BiLSTMAnalyzer
                analyzer_cls = BiLSTMAnalyzer
            elif model_i == GAMON:
                from cgsa.ml.gamon import GamonAnalyzer
                analyzer_cls = GamonAnalyzer
            elif model_i == GUENTHER:
                from cgsa.ml.guenther import GuentherAnalyzer
                analyzer_cls = GuentherAnalyzer
            elif model_i == HU_LIU:
                from cgsa.lexicon.hu_liu import HuLiuAnalyzer
                analyzer_cls = HuLiuAnalyzer
            elif model_i == JUREK:
                from cgsa.lexicon.jurek import JurekAnalyzer
                analyzer_cls = JurekAnalyzer
            elif model_i == KOLCHYNA:
                from cgsa.lexicon.kolchyna import KolchynaAnalyzer
                analyzer_cls = KolchynaAnalyzer
            elif model_i == LBA:
                from cgsa.dl.lba import LBAAnalyzer
                analyzer_cls = LBAAnalyzer
            elif model_i == MUSTO:
                from cgsa.lexicon.musto import MustoAnalyzer
                analyzer_cls = MustoAnalyzer
            elif model_i == MOHAMMAD:
                from cgsa.ml.mohammad import MohammadAnalyzer
                analyzer_cls = MohammadAnalyzer
            elif model_i == MVRNN:
                from cgsa.dl.mvrnn import MVRNNAnalyzer
                analyzer_cls = MVRNNAnalyzer
            elif model_i == RNN:
                from cgsa.dl.rnn import RNNAnalyzer
                analyzer_cls = RNNAnalyzer
            elif model_i == RNTN:
                from cgsa.dl.rntn import RNTNAnalyzer
                analyzer_cls = RNTNAnalyzer
            elif model_i == SEVERYN:
                from cgsa.dl.severyn import SeverynAnalyzer
                analyzer_cls = SeverynAnalyzer
            elif model_i == TABOADA:
                from cgsa.lexicon.taboada import TaboadaAnalyzer
                analyzer_cls = TaboadaAnalyzer
            elif model_i == YESSENALINA:
                from cgsa.dl.yessenalina import YessenalinaAnalyzer
                analyzer_cls = YessenalinaAnalyzer
            else:
                raise NotImplementedError
            self._models.append(analyzer_cls(*a_args, **a_kwargs))

    def _generate_ts(self, a_data):
        """Generate training set.

        Args:
          a_data (list): input instances

        Returns:
          2-tuple(list, list):
            lists of input features and expected classes

        """
        x, y = [], []
        if not a_data:
            return (x, y)
        for msg_i in a_data:
            if not msg_i:
                continue
            x.append(msg_i)
            y_i = msg_i.label
            # we use a pre-defined mapping of symbolic labels to integers, as
            # we need these labels to be sorted proportionally to the
            # subjective scores they get assigned when optimizing threshold of
            # lexicon-based methods
            assert y_i in CLS2IDX, "Unknown label {:s}".format(y_i)
            y_i = CLS2IDX[y_i]
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
            self._wbench = np.zeros((len(self._models), len(CLS2IDX)))
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
        self._embeddings = None

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
        a_model.reset()
        if isinstance(a_model, DLBaseAnalyzer):
            a_model.save(abspath)
        else:
            with open(abspath, "wb") as ofile:
                dump(a_model, ofile)
        self._logger.info(
            "Model %s saved to %s...",
            a_model.name, abspath
        )
        self._model_paths.append(
            os.path.relpath(abspath, a_path)
        )
        return self._model_paths[-1]
