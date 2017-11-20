#!/usr/bin/env python2.7
# -*- coding: utf-8; -*-

"""Implementation of Trie data structure

Attributes:
  SPACE_RE (re): regular expression matching continuous runs of space
    characters
  FINAL_SPACE_RE (re): regular expression matching leading and trailing white
    spaces
  ANEW (re): flag indicating that the search should start anew
  CONTINUE (re): flag indicating that the search should continue where it
    stopped
  State (class): single trie state with associated transitions
  Trie (class): implementation of Trie data structure

"""

##################################################################
# Imports
from __future__ import unicode_literals, print_function
from future.utils import python_2_unicode_compatible
from six import iteritems
import re

from cgsa.utils.common import LOGGER


##################################################################
# Variable and Constants
PUNCT_RE = re.compile("[#,.]")
SPACE_RE = re.compile("\s\s+", re.L)
FINAL_SPACE_RE = re.compile("(:?^\s+|\s+$)")
ANEW = 0
CONTINUE = 1


##################################################################
# Methods
def normalize_string(a_string, a_ignorecase=False):
    """Clean input string and return its normalized version

    @param a_string - string to clean
    @param a_ignorecase - boolean indication whether string should be
      lowercased

    @return normalized string

    """
    a_string = PUNCT_RE.sub("", a_string)
    a_string = SPACE_RE.sub(' ', a_string)
    a_string = FINAL_SPACE_RE.sub("", a_string)
    if a_ignorecase:
        a_string = a_string.lower()
    return a_string


##################################################################
# Classes
class State(object):
    """Single Trie state with associated transitions

    Instance variables:
    classes - custom classes associated with the current state
    final - boolean flag indicating whether state is final
    transitions - set of transitions triggered by the state

    Methods:
    __init__() - class constructor
    add_transition() - add new transition from that state
    check() - check transitions associated with the given character

    """

    def __init__(self, a_final=False, a_class=None):
        """
        Class constructor

        @param a_final - boolean flag indicating whether the state is
                         final or not
        @param a_class - custom class associated with the final state
        """
        # custom classes associated with the current state
        self.classes = set([])
        if a_class is not None:
            self.classes.add(a_class)
        # boolean flag indicating whether the state is final
        self.final = a_final
        # set of transitions triggered by the given state
        self.transitions = dict()

    def add_transition(self, string, pos):
        """Add new transition outgoing from this state.

        Params:
          string (str): string to be associated with transition
          pos (str or None): string or part-of-speech tag associated with
            transition

        Returns:
          address of the target state of that transition

        """
        key = (string, pos)
        if key not in self.transitions:
            self.transitions[key] = State()
        return self.transitions[key]

    def check(self, istring, ipos=None, start=-1):
        """Check transitions associated with the given input.

        Params:
          istring (str): string to be associated with transition
          ipos (str or None): string or part-of-speech tag associated with
            transition
          start (int): start of the match

        Returns:
          set of target states triggered by character

        """
        ret = set()
        # print("istring:", repr(istring))
        # print("ipos:", repr(ipos))
        # print("self.transitions:", repr(self.transitions))
        if ipos is None:
            for (string, _), trg_states in iteritems(self.transitions):
                if string == istring:
                    ret |= trg_states
            return ret
        else:
            key = (istring, ipos)
            if key in self.transitions:
                ret.add((self.transitions[key], start))
            key = (istring, None)
            if key in self.transitions:
                ret.add((self.transitions[key], start))
        return ret


@python_2_unicode_compatible
class Trie(object):
    """Implementation of trie data structure

    Attributes:
      ignorecase (bool): boolean flag indicating whether the case
                   should be ignored
      active_state (set[State]): set of currently active Trie states

    """

    def __init__(self, a_ignorecase=False):
        """Class constructor

        Args:
          a_ignorecase (bool): boolean flag indicating whether the case should
            be ignored

        """
        # boolean flag indicating whether character case should be ignored
        self.ignorecase = a_ignorecase
        self._init_state = State()
        # set of currently active Trie states
        self.active_states = set([])
        self._logger = LOGGER

    def add(self, toks, tags, a_class=0):
        """Add new string to the trie

        Args:
          toks (list[str]): string(s) to be added
          tags (list[str]): list of part-of-speech tags corresponding to string
          class (tuple): optional custom class associated with that string

        Returns:
          State:

        """
        # normalize strings
        toks = [normalize_string(itok, self.ignorecase)
                for itok in toks]
        assert len(toks) == len(tags), \
            "Unequal number of PoS tags and strings provided to" \
            " the automaton: {!r} vs. {!r}".format(toks, tags)
        # successively add states
        astate = self._init_state
        for itok, itag in zip(toks, tags):
            astate = astate.add_transition(itok, itag)
        astate.final = True
        astate.classes.add(a_class)
        return astate

    def search(self, a_input):
        """Find Trie entries in the given string.

        Args:
          a_input (list[tuple]): lemmas, forms, and tokens to be matched

        Returns:
          bool: True if at least one match succeeded

        """
        # a match state comprises information about current Trie's state, the
        # begin of the match, and the ength of the match
        result = set()
        new_states = set()
        crnt_states = set()
        for i, (form_i, lemma_i, pos_i) in enumerate(a_input):
            crnt_states.add((self._init_state, i))
            for state_j, start_j in crnt_states:
                # add match object if the state is final
                new_states |= state_j.check(form_i, pos_i, start_j)
                new_states |= state_j.check(lemma_i, pos_i, start_j)
            for state_j, start_j in new_states:
                if state_j.final:
                    result.add((state_j, start_j, i))
            crnt_states, new_states = new_states, crnt_states
            new_states.clear()
        # leave leftmost longest matches
        return [(state.classes, start, end)
                for state, start, end in self.select_llongest(result)]

    def select_llongest(self, result):
        """Find Trie entries in the given string.

        Args:
          result (set[tuple]): matched states, start, and end positions of
            matches

        Returns:
          bool: True if at least one match succeeded

        """
        # set of tuples with states and associated match objects
        result = list(result)
        result.sort(key=lambda x: (x[-2], x[-1]))
        match2delete = []
        prev_i = prev_start = prev_end = -1
        for i, (_, start_i, end_i) in enumerate(result):
            if start_i == prev_start:
                if end_i == prev_end:
                    continue
                # since the matches are sorted in ascending order, longest
                # match will necessarily be on the right
                else:
                    match2delete.append(prev_i)
            elif start_i <= prev_end:
                match2delete.append(i)
                continue
            prev_i, prev_start, prev_end = i, start_i, end_i
        for i in reversed(match2delete):
            del result[i]
        return result

    def reset(self):
        """Remove members which cannot be serialized.

        """
        self._logger = None

    def restore(self):
        """Restore members which could not be serialized.

        """
        self._logger = LOGGER

    def __str__(self):
        """Return a unicode representation of the given trie

        Returns:
          (unicode): unicode object representing the trie in graphviz format

        """
        ret = """digraph Trie {
        size="106,106";
        rankdir = "LR";
        graph [fontname="fixed"];
        node [fontname="fixed"];
        edge [fontname="fixed"];
        {node [color=black,fillcolor=palegreen,style=filled,forcelabels=true];
        """
        istate = None
        scnt = rels = ""
        state_cnt = 0
        state2cnt = {self._init_state: str(state_cnt)}
        new_states = [self._init_state]
        visited_states = set()
        while new_states:
            istate = new_states.pop()
            scnt = state2cnt[istate]
            if istate.final:
                ret += '{:s} [shape=box,fillcolor=' \
                       'lightblue,label="{:s}"];\n'.format(
                           scnt, ", ".join(
                               "({:s})".format(iclass[0])
                               for iclass in istate.classes))
            else:
                ret += "{:s} [shape=oval];\n".format(scnt)
            for (jstring, jpos), jstate in iteritems(istate.transitions):
                if jstate not in state2cnt:
                    state_cnt += 1
                    state2cnt[jstate] = str(state_cnt)
                if jstate not in visited_states:
                    new_states.append(jstate)
                rels += scnt + "->" + state2cnt[jstate] \
                    + """[label="'{:s} ({:s})'"];\n""".format(jstring,
                                                              jpos)
            visited_states.add(istate)
        ret += "}\n" + rels + "}\n"
        return ret
