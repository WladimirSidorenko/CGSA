#!/usr/bin/env python
# -*- coding: utf-8; mode: python; -*-

"""Module providing class for generic word embeddings.

Attributes:
  WEMB (class):
    class for fast retrieval and adjustment of the Google word embeddings

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function

import gc

from cgsa.utils.common import LOGGER


##################################################################
# Methods
def load_word2vec(a_fname):
    """Load Word2Vec data from file.

    Args:
      a_fname (str): file containing W2V data

    Returns:
      dict:
        mapping from word to the respective embedding

    """
    from gensim.models import KeyedVectors
    LOGGER.info("Loading %s... ", a_fname)
    w2v = KeyedVectors.load_word2vec_format(
        a_fname, binary=False, unicode_errors="replace")
    LOGGER.info("Loaded %s.", a_fname)
    return w2v


##################################################################
# Class
class LoadOnDemand(object):
    """Custom class for deferring loading of huge resources.

    Loads resources only if they are actually used.

    Attributes:
      resource (object or None): loaded resource
      cmd (method): method to load the  resource
      args (list): arguments to pass to ``cmd``
      kwargs (dict): keyword arguments to pass to ``cmd``

    """

    def __init__(self, a_cmd, *a_args, **a_kwargs):
        """Class cosntructor.

        Args:
          a_cmd (method): custom method to load the resource
          args (list): arguments to pass to ``a_cmd``
          kwargs (dict): keyword arguments to pass to ``a_cmd``

        """
        self.resource = None
        self.cmd = a_cmd
        self.args = a_args
        self.kwargs = a_kwargs
        self.__contains__ = self.__load_contains__
        self.__getitem__ = self.__load_getitem__

    def __load_contains__(self, a_name):
        """Proxy method for looking up a word in the resource.

        Args:
          a_name (str): word to look up in the resource

        Note:
          forwards the request to the underlying resource

        """
        self.load()
        return self.__noload_contains__(a_name)

    __contains__ = __load_contains__

    def __noload_contains__(self, a_name):
        """Proxy method for looking up a word in the resource.

        Args:
          a_name (str): word to look up in the resource

        Note:
          forwards the request to the underlying resource

        """
        return a_name in self.resource

    def __load_getitem__(self, a_name):
        """Proxy method for accessing the resource.

        Args:
          a_name (str): word to look up in the resource

        Note:
          forwards the request to the underlying resource

        """
        # initialize the resource if needed
        self.load()
        return self.__noload_getitem__(a_name)

    __getitem__ = __load_getitem__

    def __noload_getitem__(self, a_name):
        """Proxy method for accessing the resource.

        Args:
          a_name (str): word to look up in the resource

        Note:
          forwards the request to the underlying resource

        """
        return self.resource[a_name]

    def load(self):
        """Force loading the resource.

        Note:
          loads the resource

        """
        if self.resource is None:
            self.resource = self.cmd(*self.args, **self.kwargs)
            self.__contains__ = self.__noload_contains__
            self.__getitem__ = self.__noload_getitem__

    def unload(self):
        """Unload the resource.

        Note:
           unloads the resource

        """
        if self.resource is not None:
            resource_name = "{:s}".format(str(self.resource))
            LOGGER.info("Unloading %s...", resource_name)
            del self.resource
            self.resource = None
            gc.collect()
            self.__contains__ = self.__load_contains__
            self.__getitem__ = self.__load_getitem__
            LOGGER.info("Unloaded %s...", resource_name)


class Singleton(object):
    def __init__(self, klass):
        self.klass = klass
        self.instance = None

    def __call__(self, *args, **kwds):
        if self.instance is None:
            self.instance = self.klass(*args, **kwds)
        return self.instance


@Singleton
class Word2Vec(object):
    """Class for cached retrieval of word embeddings.

    """

    def __init__(self, path):
        """Class cosntructor.

        Args:
          path (str): path the word2vec file

        """
        self._w2v = LoadOnDemand(load_word2vec, path)
        self._cache = {}
        self.ndim = -1

    def __contains__(self, a_word):
        """Proxy method for looking up a word in the resource.

        Args:
        a_word (str): word to look up in the resource

        Returns:
        (bool):
        true if the word is present in the underlying resource

        """
        if a_word in self._cache:
            return True
        elif a_word in self._w2v:
            self._cache[a_word] = self._w2v[a_word]
            return True
        return False

    def __getitem__(self, a_word):
        """Proxy method for looking up a word in the resource.

        Args:
        a_word (str): word to look up in the resource

        Returns:
        (bool):
        true if the word is present in the underlying resource

        """
        if a_word in self._cache:
            return self._cache[a_word]
        elif a_word in self._w2v:
            emb = self._cache[a_word] = self._w2v[a_word]
            return emb
        raise KeyError

    def load(self):
        """Load the word2vec resource.

        Args:
        (void):

        Returns:
        (void):
        load the resource in place

        """
        self._w2v.load()
        self.ndim = self._w2v.resource.vector_size

    def unload(self):
        """Unload the word2vec resource.

        Args:
        (void):

        Returns:
        (void):
        load the resource in place

        """
        self._cache.clear()
        self._w2v.unload()
