#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from builtins import range
from bisect import bisect_left
from collections import Counter
from itertools import chain
from operator import mod
from six import itervalues
import pandas as pd
import os
import re

from cgsa.base import (LEX_CLMS, LEX_TYPES, NEG_SFX_RE,
                       USCORE_RE, QUOTE_NONE)
from cgsa.constants import (ENCODING, INTENSIFIERS, PUNCT_RE, SPACE_RE,
                            CLS2IDX)
from cgsa.lexicon.base import LexiconBaseAnalyzer
from cgsa.utils.common import LOGGER
from cgsa.utils.trie import Trie

##################################################################
# Variables and Constants
SKIP = {"adj": set(["selbst", "sogar", "zu", "sein", "bin", "bist", "ist",
                    "sind", "seid", "war", "warst", "wart", "waren", "wäre",
                    "wärest", "wäret", "wären", "habe", "hast", "hat",
                    "haben", "habt", "gehabt", "hätte", "hättest", "hätten",
                    "hättet", "hätte", "hättest", "hätten", "hättet",
                    "mache", "machst", "macht", "machen", "machte", "machtest",
                    "machst", "machst", "machst", "machst", "machst", "done",
                    "des", "der", "als", "ART", "PPOSAT"]),
        "adv": set(["VVFIN", "VVIMP", "VVINF", "VVIZU", "VVPP", "VAFIN",
                    "VAIMP", "VAINF", "VAPP", "VMFIN", "VMINF", "VMPP"]),
        "verb": set(["PTKZU", "sein", "bin", "bist", "ist",
                     "sind", "seid", "war", "warst", "wart", "waren", "wäre",
                     "wärest", "wäret", "wären", "habe", "hast", "hat",
                     "haben", "habt", "hätte", "hättest", "hätten", "hättet",
                     "hätte", "hättest", "hätten", "hättet"]),
        "noun": set(["PWAT", "ART", "PDAT", "PIAT", "PIDAT", "PDAT", "PPOSAT",
                     "PRELAT", "PWAT", "TRUNC", "NN", "NE", "der", "des",
                     "von", "habe", "hast", "hat", "haben", "habt", "gehabt",
                     "hätte", "hättest", "hätten", "hättet", "hätte",
                     "hättest", "hätten", "hättet", "komme", "kommst", "kommt",
                     "kommen", "komme", "kommst", "komme", "kommst", "mit",
                     "enthalten", "enthalte", "enthältst", "enthält",
                     "enthaltet", "enthaltete", "enthaltetest", "enthaltetet",
                     "enthalteten", "enthaltetet", "enthalten", "enthalten"])}
BLOCKER_CUTOFF = 0.6
HIGHLIGHTERS = {"aber": 2., "obwohl": 5.}
DETERMINERS = set(["der", "die", "das", "dem", "den",
                   "dieser", "diese", "dieses", "diesem", "diesen",
                   "PWAT", "ART", "PDAT", "PIAT", "PIDAT", "PDAT", "PPOSAT",
                   "PRELAT", "PWAT"])
IRREALIS = set(["erwarte", "erwartest", "erwartet", "erwarten",
                "erwartete", "erwartetest", "erwartetet", "erwarteten",
                "hoffe", "hoffst", "hofft", "hoffen",
                "hoffte", "hofftest", "hofftet", "hofften", "gehofft",
                "will", "willst", "wollen", "wollt", "wollte", "wolltest",
                "wollten", "wolltet", "gewollt",
                "zweifle", "zweifelst", "zweifelt", "zweifeln", "zweifelt",
                "zweifelte", "zweifeltest", "zweifelten", "zweifeltet",
                "gezweifelt",
                "denke", "denkst", "denkt", "denken",
                "dachte", "dachtest", "dachten", "dachtet", "gedacht",
                "vermute", "vermutest", "vermutet", "vermuten",
                "vermutete", "vermutetest", "vermuteten", "vermutetet",
                "annehme", "annimmst", "annimmt", "annehmen", "annehmt",
                "annahm", "annahmst", "annahmen", "annahmt", "angenommen",
                # beware, `meine' is too ambiguous and was excluded
                "meinst", "meint", "meinen",
                "meinte", "meintest", "meinten", "meintet", "gemeint",
                "vorstelle", "vorstellst", "vorstellt", "vorstellen",
                "vorstellte", "vorstelltest", "vorstellten", "vorstelltet",
                "vorgestellt",
                "könnte", "könntest", "könnten", "könntet",
                "sollte", "solltest", "sollten", "solltet",
                "würde", "würdest", "würden", "würdet",
                "wollte", "wolltest", "wollten", "wolltet",
                "möchte", "möchtest", "möchten", "möchtet",
                "müsste", "müsstest", "müssten", "müsstet",
                "etwas", "irgendetwas",
                "beliebig", "beliebige", "beliebiger", "beliebigem",
                "beliebigen", "beliebiges"])
SENT_PUNCT_RE = re.compile(r"^[.;:!?]$")
NOT_WANTED_ADJ = set(["andere", "anderen", "anderem", "anderes", "anderer",
                      "gleiche", "gleichen", "gleichem", "gleiches",
                      "gleicher",
                      "selbe", "selben", "selbem", "selbes", "selber",
                      "solche", "solchen", "solchem", "solches", "solcher",
                      "erste", "ersten", "erstem", "erstes", "erster",
                      "nächste", "nächsten", "nächstem", "nächstes",
                      "nächster",
                      "letzte", "letzten", "letztem", "letztes", "letzter",
                      "einige", "einigen", "einigem", "einiges", "einiger",
                      "viele", "vielen", "vielem", "vieles", "vieler",
                      "wenige", "wenigen", "wenigem", "weniges", "weniger",
                      "mehrere", "mehreren", "mehrerem", "mehreres",
                      "mehrerer",
                      "wenigste", "wenigsten", "wenigstem", "wenigstes",
                      "wenigster",
                      "meiste", "meisten", "meistem", "meistes", "meister"])
NOT_WANTED_ADV = set(["wirklich", "echt", "ganz",
                      "besonders", "insbesondere", "vornehmlich",
                      "hauptsächlich", "anscheinend", "offensictlich",
                      "offenbar", "scheinbar", "tatsächlich", "eigentlich",
                      "offenkundig", "plötzlich", "unvermittelt", "abrupt",
                      "komplett", "völlig", "vollständig", "vollkommen",
                      "gänzlich", "äußerst", "schlagartig", "ehrlich", "offen",
                      "grundsätzlich", "prinzipiell", "grundlegend",
                      "höchstwahrscheinlich", "wahrscheinlich", "vermutlich",
                      "wohl", "beinahe", "fast", "nahezu", "ungefähr",
                      "äußerst", "sehr", "höchst", "genau", "eben", "exakt",
                      "genauso", "ebenso", "gleichermaßen",
                      "gleichfalls", "buchstäblich", "wörtlich", "geradezu",
                      "bestimmt", "absolut", "durchaus", "definitiv",
                      "eindeutig", "praktisch", "sichtlich", "sofort",
                      "gleich", "unmittelbar", "unverzüglich", "umgehend",
                      "schleunigst", "absichtlich", "vorsätzlich", "gewollt",
                      "wissentlich", "willentlich", "normalerweise",
                      "gewöhnlich", "meist", "üblicherweise", "generell",
                      "meistens", "gewöhnlicherweise", "bald", "demnächst",
                      "gleich", "nächstens", "offensichtlich", "deutlich",
                      "klar", "deutlich", "sicherlich", "mild", "sanft",
                      "zufällig", "versehentlich", "zufälligerweise",
                      "schließlich", "letztendlich", "endlich", "persönlich",
                      "wichtig", "zudem", "speziell", "ausdrücklich",
                      "gezielt", "eigens", "voraussichtlich", "unbedingt",
                      "absolut", "durchaus", "völlig", "total", "wirklich",
                      "zwangsläufig", "zwingend", "notwendigerweise",
                      "zwingendermaßen", "dringend", "stark", "kräftig",
                      "relativ", "vergleichsweise", "verhältnismäßig",
                      "ganz", "gänzlich", "möglich", "möglicherweise",
                      "eventuell", "vielleicht", "allgemein", "gewöhnlich",
                      "üblicherweise", "überhaupt", "hauptsächlich",
                      "generell", "ausdrücklich", "eigens", "insbesondere",
                      "letztlich", "endlich", "ursprünglich",
                      "zuerst", "zunächst", "anfänglich", "anfangs", "quasi",
                      "nahezu", "praktisch", "geradezu", "gewissermaßen",
                      "offen", "unumwunden", "ernst", "ernsthaft", "ziemlich",
                      "recht", "ordentlich", "etwa", "circa", "zirka",
                      "ungefähr", "kritisch", "entscheidend", "andauernd",
                      "kontinuierlich", "dauernd", "stets", "unablässig",
                      "fortwährend", "natürlich", "sicher", "gewiss",
                      "bestimmt", "regulär", "regelmäßig", "eigentlich",
                      "essentiell", "neulich", "kürzlich", "neuerdings",
                      "unlängst", "ausdrücklich", "deutlich", "explizit",
                      "genau", "gerade", "sehr", "geschickt", "fein",
                      "zuletzt", "mündlich", "technisch", "technologisch",
                      "erstens", "idealerweise", "menschlich", "sexuell",
                      "sexual", "gesellschaftlich", "sozial", "vorzugsweise",
                      "möglichst", "legal", "gesetzlich", "rechtlich",
                      "juristisch", "hoffentlich", "oft", "öfters", "häufig",
                      "oftmalig", "oftmals", "größtenteils", "weitgehend",
                      "überwiegend", "großteils", "sachlich", "objektiv",
                      "typisch", "normalerweise", "typischerweise",
                      "üblicherweise", "auch"])
CMP_COEFF = INTENSIFIERS.loc[INTENSIFIERS["intensifier"] == "mehr"].score.max()
PRIMARY_LABEL_SCORE = 0.51
SECONDARY_LABEL_SCORES = (1. - PRIMARY_LABEL_SCORE) / float(len(CLS2IDX) - 1)


##################################################################
# Classes
class TaboadaAnalyzer(LexiconBaseAnalyzer):
    """Class for lexicon-based sentiment analysis.

    Attributes:

    """

    class PolTermMatches(object):
        """Class comprising matches pertaining to specific parts of speech.

        """
        def __init__(self):
            """Class constructor."""
            self.adjectives = []
            self.adverbs = []
            self.nouns = []
            self.verbs = []
            # odered mapping from part-of-speech tags to corresponding
            # containers for matches
            self.TAGS2CONTAINER = [
                (set(["VAFIN", "VAIMP", "VAIMP", "VAINF", "VAPP",
                      "VMFIN", "VMINF", "VVFIN", "VVIMP", "VVINF",
                      "VVIZU", "VVPP"]), self.verbs),
                (set(["NE", "NN", "FM", "XY"]), self.nouns),
                (set(["ADJA", "ADJD"]), self.adjectives)
                # adverbs will be used by default for all remaining cases
            ]

        def __repr__(self):
            ret = ("<{:s}: adjectives: {!r}; adverbs: {!r};"
                   " nouns: {!r}; verbs: {!r}>").format(
                       self.__class__.__name__,
                       self.adjectives, self.adverbs,
                       self.nouns, self.verbs)
            return ret

    def __init__(self, lexicons=[]):
        """Class constructor.

        Args:
          lexicons (list[str]): list of sentiment lexicons

        """
        assert lexicons, \
            "Provide at least one lexicon for lexicon-based method."
        super(TaboadaAnalyzer, self).__init__()
        self.name = "taboada"
        self._logger = LOGGER
        self._read_lexicons(self._polar_terms, lexicons, a_encoding=ENCODING)
        self._use_abs_so = False
        self._use_cnt_so = False
        self._use_mean_so = False
        self._thresholds = None

    def train(self, train_x, train_y, dev_x, dev_y,
              **kwargs):
        # no training is required for this method
        scores = [self._compute_so(tweet_i)
                  for tweet_i in chain(train_x, dev_x)]
        labels = [label_i
                  for label_i in chain(train_y, dev_y)]
        abs_scores = []
        cnt_scores = []
        mean_scores = []
        for abs_score_i, count_i, mean_score_i in scores:
            abs_scores.append(abs_score_i)
            cnt_scores.append(count_i)
            mean_scores.append(mean_score_i)
        # check thresholds for every score type
        f1_abs, abs_thresholds = self._optimize_thresholds(
            abs_scores, labels
        )
        f1_cnt, cnt_thresholds = self._optimize_thresholds(
            cnt_scores, labels
        )
        f1_mean, mean_thresholds = self._optimize_thresholds(
            mean_scores, labels
        )
        self._logger.info("f1_abs: %f; f1_cnt: %f; f1_mean: %f",
                          f1_abs, f1_cnt, f1_mean)
        best_f1 = f1_abs
        if f1_abs > f1_cnt:
            if f1_abs > f1_mean:
                self._use_abs_so = True
                self._thresholds = abs_thresholds
            else:
                best_f1 = f1_mean
                self._use_mean_so = True
                self._thresholds = mean_thresholds
        else:
            if f1_cnt > f1_mean:
                best_f1 = f1_cnt
                self._use_cnt_so = True
                self._thresholds = cnt_thresholds
            else:
                best_f1 = f1_mean
                self._use_mean_so = True
                self._thresholds = mean_thresholds
        self._logger.info("best F1: %f; self._thresholds: %r;",
                          best_f1, self._thresholds)

    def predict_proba(self, msg, yvec):
        """Method for predicting sentiment propbablities of a single message.

        Args:
          msg (cgsa.utils.data.Tweet):
            discourse relation whose sense should be predicted
          yvec (np.array): target array for storing the probabilities

        Returns:
          void

        Note:
          modifies `'

        """
        total_so, total_cnt, avg_so = self._compute_so(msg)
        if self._use_abs_so:
            score = total_so
        elif self._use_cnt_so:
            score = total_cnt
        else:
            score = total_cnt
        label = bisect_left(self._thresholds, score)
        self._logger.debug("score: %f, label: %d, yvec: %r;",
                           score, label, yvec)
        yvec[:] = SECONDARY_LABEL_SCORES
        yvec[label] = PRIMARY_LABEL_SCORE
        self._logger.debug("resulting yvec: %r;", yvec)

    def _compute_so(self, tweet):
        """Compute semantic orientation of a tweet.

        Args:
          tweet (cgsa.utils.data.Tweet): input message

        Returns:
          tuple: total SO score, total SO count, average SO value

        """
        total_so = 0.
        total_cnt = 0
        orig_forms = [w_i.form for w_i in tweet]
        forms = [self._preprocess(form_i) for form_i in orig_forms]
        lemmas = [self._preprocess(w_i.lemma) for w_i in tweet]
        tags = [w_i.tag for w_i in tweet]
        feats = [w_i.feats for w_i in tweet]
        assert len(forms) == len(lemmas), \
            "Unmatching number of tokens and lemmas."
        assert len(forms) == len(tags), "Unmatching number of tokens and tags."

        def join_matches(matches):
            """Sum scores of all matches."""
            return [(sum(res[-1] for res in results), start, end)
                    for results, start, end in matches]

        match_input = [(f, l, t) for f, l, t in zip(forms, lemmas, tags)]
        # match all polar terms
        polterm_matches = join_matches(self._polar_terms.search(match_input))
        # split matches according to their leading parts of speech
        polterm_matches = self._split_polterm_matches(tags, polterm_matches)
        self._logger.debug("matched polar terms: %r", polterm_matches)
        # match intensifiers (note: we are only interested in the last and
        # first positions of the matches, with the possibility to lookup the
        # start by the end position)
        int_matches = {end: (start, score)
                       for score, start, end
                       in self._intensifiers.search(match_input)}
        self._logger.debug("matched intensifiers: %r", int_matches)
        # match negations
        neg_matches = [(start, end)
                       for _, start, end
                       in self._negations.search(match_input)]
        self._logger.debug("matched negations: %r", neg_matches)
        # match blocking constructs
        boundaries = self._find_boundaries(match_input)
        self._logger.debug("boundaries: %r", boundaries)
        # compute semantic orientations (SO) of nouns
        noun_so, noun_cnt = self._compute_terms_so(
            forms, lemmas, tags, feats,
            polterm_matches.nouns, int_matches,
            neg_matches, boundaries, "noun"
        )
        self._logger.debug("noun_so: %r; noun_cnt: %r;",
                           noun_so, noun_cnt)
        # compute semantic orientations of verbs
        verb_so, verb_cnt = self._compute_terms_so(
            forms, lemmas, tags, feats,
            polterm_matches.nouns, int_matches,
            neg_matches, boundaries, "verb"
        )
        self._logger.debug("verb_so: %r; verb_cnt: %r;",
                           verb_so, verb_cnt)
        # compute semantic orientations of adjectives
        adj_so, adj_cnt = self._compute_terms_so(
            forms, lemmas, tags, feats,
            polterm_matches.adjectives, int_matches,
            neg_matches, boundaries, "adj"
        )
        self._logger.debug("adj_so: %r; adj_cnt: %r;",
                           adj_so, adj_cnt)
        # compute semantic orientations of adverbs
        adv_so, adv_cnt = self._compute_terms_so(
            forms, lemmas, tags, feats,
            polterm_matches.adverbs, int_matches,
            neg_matches, boundaries, "adv"
        )
        self._logger.debug("adv_so: %r; adv_cnt: %r;",
                           adv_so, adv_cnt)
        total_so = noun_so + verb_so + adj_so + adv_so
        total_cnt = noun_cnt + verb_cnt + adj_cnt + adv_cnt
        avg_so = total_so / (float(total_cnt) or 1e10)
        return total_so, total_cnt, avg_so

    def _compute_terms_so(self,
                          forms, lemmas, tags, feats,
                          term_matches, intensifiers,
                          negations, boundaries, pos):
        """Compute semantic orientation of nouns, verbs, adjectives, and adverbs.

        Args:
          forms (list[str]): tokens of the analyzed message
          lemmas (list[str]): lemmas of the analyzed message
          tags (list[str]): tags of the analyzed message
          feats (list[dict]): token features
          term_matches (list[tuple]): matches of polar terms
          intensifiers (dict): intensifier matches
          negations (list[tuple]): negation matches
          boundaries (list[tuple]): boundary matches
          pos (str): part-of-speech of the analyzed term

        Returns:
          2-tuple: overall SO score and count of polar terms

        """
        total_score = 0.
        counts = Counter()
        term_indices = [(start, end) for _, start, end in term_matches]
        for score_i, start_i, end_i in term_matches:
            prev_pos = start_i - 1
            int_score = neg_int_score = 0.
            term_form = ' '.join(forms[start_i:end_i + 1])
            term_lemma = ' '.join(forms[start_i:end_i + 1])
            left_boundary = self._find_next_boundary(start_i,
                                                     boundaries)
            # check the comparison degree of adjectives
            if pos == "adj":
                if term_form in NOT_WANTED_ADJ or term_lemma in NOT_WANTED_ADJ:
                    continue
                skip_term = False
                for j in range(start_i, end_i + 1):
                    if feats[j].get("comp"):
                        int_score += CMP_COEFF
                        skip_term = not (self._check4predicate(left_boundary,
                                                               start_i, tags)
                                         and self._check4determiners(
                                             start_i, forms, lemmas, tags, 2))
                        break
                    elif feats[j].get("sup"):
                        int_score += 1.
                        skip_term = not self._check4predicate(left_boundary,
                                                              start_i, tags)
                        break
                if skip_term:
                    continue
                # look past determiners and "as" for intensification
                if lemmas[prev_pos] == "als" or self._check4determiners(
                        prev_pos, forms, lemmas, tags):
                    self._logger.debug(
                        "skipping determiner at position %d while computing "
                        "adjective intensifier", prev_pos)
                    prev_pos -= 1
                while prev_pos in intensifiers:
                    # use each intensifier only once
                    int_start, int_score = intensifiers.pop(prev_pos)
                    if forms[int_start] and forms[int_start][0].isupper():
                        int_score *= 2  # `capital_modifier` in Taboada et al.
                        prev_pos = int_start - 1
            else:
                if pos == "adv":
                    if term_form in NOT_WANTED_ADV \
                       or term_lemma in NOT_WANTED_ADV:
                        continue
                    if lemmas[prev_pos] == "als":
                        prev_pos -= 1
                if prev_pos in intensifiers:
                    # use each intensifier only once
                    int_start, int_score = intensifiers.pop(prev_pos)
                    if forms[int_start] and forms[int_start][0].isupper():
                        int_score *= 2  # `capital_modifier` in Taboada et al.
                    prev_pos = int_start - 1
                if pos == "verb":
                    right_boundary = self._find_next_boundary(end_i,
                                                              boundaries,
                                                              left=False)
                    if right_boundary > 0:
                        right_boundary -= 1
                    if right_boundary in intensifiers:
                        # this is apparently a bug, but we need to reproduce
                        # Taboada's method as is
                        istart, int_score = intensifiers.pop(
                            right_boundary
                        )
            # determine tokens, which come into consideration as negation
            neg_start, neg_end = self._find_negation(
                prev_pos, negations, boundaries,
                forms, lemmas, tags, pos
            )
            # determine negation intensification
            if neg_start >= 0:
                prev_pos = neg_start - 1
                skip = SKIP["adj"]
                while prev_pos >= 0 and (forms[prev_pos] in skip
                                         or lemmas[prev_pos] in skip
                                         or tags[prev_pos] in skip):
                    prev_pos -= 1
                if prev_pos in intensifiers:
                    neg_int_start, neg_int_score = intensifiers.pop(prev_pos)
            # apply intensification
            if int_score:
                score_i *= 1. + int_score
            else:
                # look for a blocker up to the next boundary
                if self._find_blocker(score_i, left_boundary, start_i, pos,
                                      term_indices, term_matches,
                                      forms, lemmas, tags):
                    self._logger.debug("semantic orientation blocked")
                    score_i = 0
            if neg_start >= 0:
                if score_i < 0:
                    neg_shift = abs(score_i)
                else:
                    neg_shift = 0.8
                if score_i > 0:
                    score_i -= neg_shift
                elif score_i < 0:
                    score_i += neg_shift
                if neg_int_score:
                    score_i *= 1. + neg_int_score
            # apply other modifiers (uppercase, irrealis, quotes, etc)
            score_i = self._apply_modifiers(score_i, start_i, prev_pos,
                                            forms, lemmas, tags, boundaries)
            if score_i:
                cnt_key = ' '.join(lemmas[start_i:end_i + 1])
                counts[cnt_key] += 1
                # `use_word_counts_lower' from Taboada et al.
                score_i /= float(counts[cnt_key])
            total_score += score_i
        total_count = sum(itervalues(counts))
        return (total_score, total_count)

    def _apply_modifiers(self, score, pol_term_index, right_edge,
                         forms, lemmas, tags, boundaries):
        """Check for additional modying elements.

        Args:
          score (float): score of the polar term
          pol_term_index (int): index of the polar term
          right_edge (int): right-most position of the polar term context
          forms (list[str]): original tweet tokens
          lemmas (list[str]): lemmas of tweet tokens
          tags (list[str]): tags of the analyzed message
          boundaries (list[int]): boundary tokens

        Returns:
          float: (possibly modified) score

        """
        snt_punct = self._get_sent_punct(pol_term_index, forms, boundaries)
        if forms[pol_term_index].isupper():
            score *= 2.  # capital modifier
        if '!' in snt_punct:
            score *= 2.  # exclam modifier
        boundary = self._find_next_boundary(right_edge,
                                            boundaries) + 1
        highlighter = self._get_sent_highlighter(boundary,
                                                 right_edge + 1,
                                                 lemmas)
        if highlighter:
            self._logger.debug("score increased by highlighter")
            score *= HIGHLIGHTERS[highlighter]
        if '?' in snt_punct and not self._check4determiners(
                right_edge, forms, lemmas, tags):
            self._logger.debug("semantic orientation blocked by question mark")
            score = 0
        if self._is_in_quotes(pol_term_index, forms):
            self._logger.debug("semantic orientation blocked by quotes")
            score = 0
        if self._has_sent_irrealis(boundary, right_edge,
                                   forms, lemmas, tags):
            self._logger.debug("semantic orientation blocked by irrealis")
            score = 0
        return score

    def _check4determiners(self, index, forms, lemmas, tags, window=1):
        """Check whether given context contains determiners.

        Args:
          index (int): index to start the search from
          forms (list[str]): original tweet tokens
          lemmas (list[str]): lemmas of tweet tokens
          tags (list[str]): tags of the analyzed message
          window (int): length of the context to search for determiner

        Returns:
          bool: True if determiner was found

        """
        for i in range(index, index - window):
            if (forms[i] in DETERMINERS
                    or lemmas[i] in DETERMINERS
                    or tags[i] in DETERMINERS):
                return True
        return False

    def _check4predicate(self, start, end, tags):
        """Check whether given context contains determiners.

        Args:
          start (int): index to start the search from
          end (int): original tweet tokens
          tags (list[str]): tags of the analyzed message

        Returns:
          bool: True if determiner was found

        """
        for i in range(end, max(-1, start - 1)):
            if tags[i].startswith():
                return True
        return False

    def _find_boundaries(self, match_input):
        """Determine boundaries which block propagation.

        Args:
          match_input (list[tuple]): list of tuples comprising forms, lemmas,
            and part-of-speech tags

        Returns:
          list[tuple]: indices of matched boundaries

        """
        boundaries = self._boundaries.search(match_input)
        for i, (tok_i, _, _) in enumerate(match_input):
            if PUNCT_RE.search(tok_i):
                boundaries.append((None, i, i))
        boundaries = [(start, end)
                      for _, start, end
                      in self._boundaries.select_llongest(boundaries)]
        return boundaries

    def _find_negation(self, index,
                       neg_matches, boundaries,
                       forms, lemmas, tags, word_type):
        """Look for negations appearing in the nearby context.

        Args:
          index (int): index of the potentially negated word
          neg_matches (list[tuple]): negation matches
          boundaries (list[tuple]): boundary matches
          forms (list[str]): tokens of the analyzed message
          lemmas (list[str]): lemmas of the analyzed message
          tags (list[str]): tags of the analyzed message
          word_type (str): part of speech of the potentially negated
            word

        Returns:
          bool: true if negation was found

        """
        not_found = (-1, -1)
        search_idx = (index, index)
        # check if there is any negation preceding the polar term
        neg_idx = bisect_left(neg_matches, search_idx) - 1
        if neg_idx < 0:
            return not_found
        neg_start, neg_end = neg_matches[neg_idx]
        # check if there is a boundary between polar term and negation
        boundary = self._find_next_boundary(index, boundaries)
        if boundary > neg_end:
            self._logger.debug("negation prevented by blocking")
            return not_found
        # otherwise, check if every token between the negtion and the polar
        # term can be skipped
        skip_items = SKIP["noun"]
        for i in range(neg_end + 1, index):
            for item_i in (forms[i], lemmas[i], tags[i]):
                if item_i in skip_items:
                    continue
            self._logger.debug(
                "negation prevented by non-skipped term: %r, %r, %r",
                forms[i], lemmas[i], tags[i]
            )
            return not_found
        return (neg_start, neg_end)

    def _find_blocker(self, score, start, end, wtype,
                      pol_term_indices, pol_term_matches,
                      forms, lemmas, tags):
        """Find terms of opposite polarities that might block polar terms.

        Args:
          score (float): score of the polar term
          start (int): position of the next blocker on the left
          end (int): position of the respective polar term
          wtype (str): PoS type of the polar term
          pol_term_indices (list[tuple]): indices of polar terms
          pol_term_matches (list[tuple]): matches of polar terms
          forms (list[str]): tokens of the analyzed tweets
          lemmas (list[str]): lemmas of the analyzed tweets
          tags (list[str]): PoS tags of the analyzed tweets

        Returns:
          bool: True if a blocker was found in the nearby context

        """
        # start looking for a blocker from the preceding position
        end -= 1

        def check_scores(term_score, block_score):
            return abs(term_score + block_score) < \
                abs(term_score) + abs(block_score)

        def check_pol_term(index):
            pol_term_idx = bisect_left(pol_term_indices, (index, index))
            if pol_term_idx < len(pol_term_indices):
                score, pol_term_start, pol_term_end = \
                    pol_term_matches[pol_term_idx]
                if pol_term_start <= index <= pol_term_end:
                    return score
            return None

        skip_items = SKIP[wtype]
        for i in range(end, start, -1):
            block_score = check_pol_term(i)
            if block_score is not None and abs(block_score) >= BLOCKER_CUTOFF:
                if check_scores(score, block_score):
                    return True
            for item_i in (forms[i], lemmas[i], tags[i]):
                if item_i in skip_items:
                    break
            else:
                break
            i -= 1
        return False

    def _find_next_boundary(self, index, boundaries, left=True):
        """Find nearest boundary on the left

        Args:
          index (int): position to find the nearest boundary for
          boundaries (list[tuple]): positions of boundaries
          left (bool): find left-most boundary (otherwise, right-most boundary
            is searched)

        Return:
          int: position of the nearest boundary on the left from index

        """
        boundary_idx = bisect_left(boundaries, (index, index))
        if boundary_idx < len(boundaries):
            boundary_start, boundary_end = boundaries[boundary_idx]
            if boundary_start <= index <= boundary_end:
                return index
        if left:
            boundary_idx -= 1
        if 0 <= boundary_idx < len(boundaries):
            _, boundary_end = boundaries[boundary_idx]
            return boundary_end
        return -1

    def _get_sent_highlighter(self, start, end, lemmas):
        """Find items that might increase polar score.

        Args:
          start (int): starting position for search
          end (int): end position for search
          lemmas (list[str]): lemmas of tweet tokens

        Returns:
          str: found highlighter or empty string

        """
        for i in range(start, end):
            if lemmas[i] in HIGHLIGHTERS:
                return lemmas[i]
        return ""

    def _get_sent_punct(self, index, forms, boundaries):
        """Find closest punctuation mark on the right from index.

        Args:
          index (int): index of the polar term
          forms (list[str]): original tweet tokens
          boundaries (list[int]): boundary tokens

        Returns:
          str: closest punctuation mark on the right or empty string

        """
        idx = bisect_left(boundaries, (index, index))
        for boundary in boundaries[idx:]:
            tok = forms[boundary[0]]
            if SENT_PUNCT_RE.match(tok):
                return tok
        return ""

    def _has_sent_irrealis(self, left, right, forms, lemmas, tags):
        """Returns true if there is an irrealis marker.

        Args:
          left (int): left boundary of the potential irrealis region
          right (int): right boundary of the potential irrealis region
          forms (list[str]): tokens of the analyzed message
          lemmas (list[str]): lemmas of the analyzed message
          tags (list[str]): tags of the analyzed message

        Returns:
          bool: True if irrealis marker was found

        """
        if self._check4determiners(right, forms, lemmas, tags):
            return False
        for i in range(right, max(-1, left - 1), -1):
            if forms[i] in IRREALIS or lemmas[i] in IRREALIS:
                return True
        return False

    def _is_in_quotes(self, index, toks):
        """Check whether given token is within a quoted passsage.

        Args:
          index (int): index which should be checked
          toks (list[str]): tokens of the original tweet

        Returns:
          bool: Trie if the index is within a quaoted passage

        """
        found = False
        quotes_left = 0
        quotes_right = 0
        current = ""
        for tok_i in toks[:index + 1]:
            if SENT_PUNCT_RE.match(tok_i):
                break
            if tok_i == '"' or tok_i == "'":
                quotes_left += 1
        if mod(quotes_left, 2) == 1:
            i = index
            for i, tok_i in enumerate(toks[index + 1:]):
                if SENT_PUNCT_RE.match(tok_i):
                    break
                if tok_i == '"' or tok_i == "'":
                    quotes_right += 1
            if (quotes_left - quotes_right == 1) and i < len(toks) - 1 \
               and toks[i+1] == '"':
                quotes_right += 1
            if mod(quotes_right, 2) == 1:
                return True
        return False

    def _read_lexicons(self, a_term2polscore, a_lexicons, a_encoding=ENCODING):
        """Load lexicons.

        Args:
          a_term2polscore (cgsa.utils.trie.Trie): mapping from terms to
            their polarity scores
          a_lexicons (list): tags of the input instance
          a_encoding (str): input encoding

        Returns:
          void:

        Note:
          populates `a_pos_term2polscore` and `a_neg_term2polscore` in place

        """
        for lexpath_i in a_lexicons:
            lexname = os.path.splitext(os.path.basename(
                lexpath_i
            ))[0]
            LOGGER.debug(
                "Reading lexicon %s...", lexname
            )
            lexicon = pd.read_table(lexpath_i, header=None, names=LEX_CLMS,
                                    dtype=LEX_TYPES, encoding=a_encoding,
                                    error_bad_lines=False, warn_bad_lines=True,
                                    keep_default_na=False, na_values=[''],
                                    quoting=QUOTE_NONE)
            for i, row_i in lexicon.iterrows():
                term = USCORE_RE.sub(' ', row_i.term)
                if NEG_SFX_RE.search(term):
                    # Taboada's method explicitly accounts for negations, so we
                    # skip negated entries from the lexicon altogether
                    continue
                term = self._preprocess(term)
                a_term2polscore.add(SPACE_RE.split(term),
                                    SPACE_RE.split(row_i.pos),
                                    (lexname, row_i.polarity, row_i.score))
            LOGGER.debug(
                "Lexicon %s read...", lexname
            )

    def _split_polterm_matches(self, tags, term_matches):
        """Separate polterm matches according to their leading parts of speech.

        Args:
          tags (list[str]): PoS tags of tweet
          polterm_matches (list[tuple]): matches of polar terms

        Returns:
          PolTermMatches: matches separated by parts of speech

        """
        ret = self.PolTermMatches()
        for match_i in term_matches:
            score_i, start_i, end_i = match_i
            tags_i = set(tags[start_i:end_i + 1])
            for tags_j, contaier in ret.TAGS2CONTAINER:
                if tags_i & tags_j:
                    contaier.append(match_i)
                    break
            else:
                ret.adverbs.append(match_i)
        return ret
