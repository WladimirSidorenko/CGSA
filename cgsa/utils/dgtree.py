#!/usr/bin/env python
# -*- mode: python; coding: utf-8 -*-

##################################################################
# Documentation
"""Module providing a class for analyzing dependency trees.

"""

##################################################################
# Imports
from __future__ import absolute_import, print_function, unicode_literals


##################################################################
# Class
class DGTree(object):
    """Dependency tree.

    """
    def __init__(self, parent=None, deprel="", i=-1, children=None, deps=None):
        """Class constructor.

        Args:
          parent (list[tuple]): list of dependencies
          deprel (str): dependency relation to the parent
          deps (list[tuple] or None): list of dependencies

        """
        self.i = i             # word's index in sentence
        self.parent = parent
        self.deprel = deprel
        self.children = [] if children is None else children
        if deps is not None:
            self._parse(deps)

    def __len__(self):
        """Special method for determining length of a node.

        The length is computed as the number of ancestors +1.

        """
        return 1 + sum(len(child)
                       for child in self.children)

    def _parse(self, deps):
        """Parse input string and populate dependencies.

        Args:
          deps (list[tuple]): list of dependencies

        Returns:
          void:

        """
        # first, intialize orphan nodes
        nodes = [DGTree(parent=None, deprel=deprel, i=i)
                 for i, (_, deprel) in enumerate(deps)]
        # second, intialize parents and children
        for node, (prnt_idx, _) in zip(nodes, deps):
            if prnt_idx >= 0:
                node.parent = nodes[prnt_idx]
                nodes[prnt_idx].children.append(node)
            else:
                self.children.append(node)
