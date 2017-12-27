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
from six.moves import range

from cgsa.constants import INTENSIFIERS
from cgsa.base import SENT_PUNCT_RE
from cgsa.lexicon.base import (LexiconBaseAnalyzer,
                               PRIMARY_LABEL_SCORE,
                               SECONDARY_LABEL_SCORE,
                               SKIP)

##################################################################
# Variables and Constants
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


##################################################################
# Classes
class TaboadaAnalyzer(LexiconBaseAnalyzer):
    """Lexicon-based sentiment analysis using the SO-CAL method of Taboada et al.

    Attributes:

    """
    def __init__(self, lexicons=[], **kwargs):
        """Class constructor.

        Args:
          lexicons (list[str]): list of sentiment lexicons

        """
        super(TaboadaAnalyzer, self).__init__(lexicons)
        self.name = "taboada"
        self._use_abs_so = False
        self._use_cnt_so = False
        self._use_mean_so = False

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
            score = avg_so
        self._logger.debug("score: %f, thresholds: %r;",
                           score, self._thresholds)
        label = bisect_left(self._thresholds, score)
        self._logger.debug("score: %f, label: %d, yvec: %r;",
                           score, label, yvec)
        yvec[:] = SECONDARY_LABEL_SCORE
        yvec[label] = PRIMARY_LABEL_SCORE
        self._logger.debug("resulting yvec: %r;", yvec)

    def _compute_so(self, tweet):
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

        match_input = [(f, l, t) for f, l, t in zip(forms, lemmas, tags)]
        # match all polar terms
        polterm_matches = self._join_scores(
            self._polar_terms.search(match_input))
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
            polterm_matches.verbs, int_matches,
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
        self._logger.debug("total_so: %f; total_cnt: %f; avg_so: %f",
                           total_so, total_cnt, avg_so)
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
                    self._logger.debug(
                        "Adjective %r listed in NOT_WANTED_ADJ (skipping)",
                        term_form
                    )
                    continue
                skip_term = False
                for j in range(start_i, end_i + 1):
                    if feats[j].get("comp"):
                        int_score += CMP_COEFF
                        skip_term = not self._check4predicate(left_boundary,
                                                              start_i, tags)
                        break
                    elif feats[j].get("sup"):
                        int_score += 1.
                        skip_term = not (self._check4predicate(left_boundary,
                                                               start_i, tags)
                                         and self._check4determiners(
                                             start_i, forms, lemmas, tags, 2))
                        break
                self._logger.debug(
                    "skip term (%r) = %r", term_form, skip_term
                )
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
        """Check for additional modyfing elements.

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
            self._logger.debug("applying capital modifier")
        if '!' in snt_punct:
            score *= 2.  # exclam modifier
            self._logger.debug("applying exclamation modifier")
        boundary = self._find_next_boundary(right_edge,
                                            boundaries) + 1
        highlighter = self._get_sent_highlighter(boundary,
                                                 right_edge + 1,
                                                 lemmas)
        if highlighter:
            self._logger.debug("score increased by highlighter")
            score *= HIGHLIGHTERS[highlighter]
            self._logger.debug("applying highlighter")
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
        for i in range(end, start, -1):
            if tags[i].startswith('V'):
                return True
        return False

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
        quotes_left = 0
        quotes_right = 0
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
