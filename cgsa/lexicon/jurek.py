#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from cgsa.lexicon.base import CondProbLexiconBaseAnalyzer
from cgsa.constants import COND_PROB_FILE
import numpy as np

##################################################################
# Variables and Constants
P = 3.5


##################################################################
# Classes
class JurekAnalyzer(CondProbLexiconBaseAnalyzer):
    """Lexicon-based sentiment analysis method of Jurek et al.

    """

    def __init__(self, lexicons=[], cond_probs=COND_PROB_FILE):
        """Attributes:


        Notes:

          Besides the sentiment value, for each word from the lexicon we
          estimated a conditional probability of the positive and negative
          classes conditioned on the occurrence of this word.  Based on a set
          of labelled data, for each positive word we estimated the probability
          that a random message containing this word is positive. In the same
          manner the probabilities were estimated for each negative word.  For
          the purpose of calculating the probabilities we applied a training
          data set provided by Stanford [29] that contains 1.6 million
          (including 800,000 positive and 800,000 negative) labelled tweets.

          Negation:

            F(S) = \max{\frac{S + 100}{2}, 10} if S < 0
            F(S) = \max{\frac{S - 100}{2}, -10} if S > 0

          Once a negation is recognised in a sentence, the first non-neutral
          word that occurs within the following three positions after the
          negator is searched.

          A sentence is classified as positive if the total sentiment is
          greater than 25.

          Intensifiers:

          Intensifiers refer to words such as very, quite, most, etc.  These
          are the words that change sentiment of the neigh- bouring non-neutral
          terms. They can be divided into two categories [29], namely
          amplifiers (very, most) and down- toners (slightly) that increase and
          decrease the inten- sity of sentiment, respectively. In our approach
          25 most frequently applied intensifiers were selected and then,
          depending on their polarity, they were divided into 3 cat- egories,
          namely downtoners, weak amplifiers and strong amplifiers. Empirically
          downtoners represent intensifiers that decrease value of the
          sentiment by 50 %. Weak and strong amplifiers increase sentiment by
          50 and 100 %, respectively.  None of the negators and intensifiers is
          included in the sentiment lexicon. Consequently, if they appear in a
          sentence surrounded by only neutral text, they are con- sidered as
          neutral words. However, if they appear in a neighbourhood of positive
          or negative words they are considered as non-neutral given that they
          influence the final sentiment of a sentence.

          Following the aforementioned evaluation, the formulas for calculating
          the overall positive and negative sentiment of a sentence were
          written as

            F_P = \min(\frac{A_P}{2 - \log(3.5\times W_P + I_P)}, 100),
            F_N = \max(\frac{A_N}{2 - \log(3.5\times W_N + I_N)}, -100);

          where I P and I N stand for the number of intensifiers that refer
          respectively to positive and negative words in a sentence. Instead of
          decreasing or increasing values of word's sentiment by 50 or 100 %,
          we simply decrease or increase the number of words by appropriate
          values of 0.5 or 1, respectively.

          The final Sentiment function validates the value of F P / F N and e P
          /e N . Depending on if the absolute value of the sentiment is greater
          than 25 or the absolute value of the evidence is higher than 0.5, it
          returns the sentiment or 0. If there are only positive words in the
          message, the final value of the sentiment is selected based on F P
          and e P only. The same happens if there are only negative words in
          the message. In case when there is a mixture of positive and negative
          words, the message is classify as positive or negative, depending on
          which, positive or negative, words are stronger. First, the
          difference between positive and negative evidence is calculated. If
          one piece of the evidence is much higher than the other (greater than
          0.1) then the positive or negative sentiment is returned,
          respectively. In case when there is no evidence available or they do
          not differ strongly enough from each other, the final decision is
          made based on the difference between positive and negative
          sentiment. If the positive sentiment is greater than the negative
          sentiment the sentence is classify as positive and vice versa.

        """
        self._cond_probs = self._load_cond_probs(cond_probs)
        super(JurekAnalyzer, self).__init__(lexicons)
        self.name = "jurek"

    def _compute_so(self, tweet):
        forms = [self._preprocess(w_i.form)
                 for w_i in tweet]
        lemmas = [self._preprocess(w_i.lemma)
                  for w_i in tweet]
        tags = [w_i.tag for w_i in tweet]
        match_input = [(f, l, t)
                       for f, l, t
                       in zip(forms, lemmas, tags)]
        # match polar term
        polterm_matches = self._join_scores(
            self._polar_terms.search(match_input))
        self._logger.debug("polterm_matches: %r;", polterm_matches)
        # match negations
        neg_matches = set([end
                           for _, _, end
                           in self._negations.search(match_input)])
        self._logger.debug("matched negations: %r", neg_matches)
        # match intensifiers
        int_matches = set([end
                           for _, _, end
                           in self._intensifiers.search(match_input)])
        self._logger.debug("matched intensifiers: %r", int_matches)
        # compute statistics on polar terms
        pos_so = neg_so = 0.
        pos_cnt = neg_cnt = 0
        pos_int_cnt = neg_int_cnt = 0
        for score_i, start_i, end_i in polterm_matches:
            # adjust scores of negated terms
            for idx in range(max(0, start_i - 3, start_i)):
                if idx in neg_matches:
                    if score_i < 0:
                        score_i = max((score_i + 100.)/2., 10.)
                    else:
                        score_i = min((score_i - 100.)/2., -10.)
                    break
            else:
                # count intensifiers if the term is not negated
                for idx in range(max(0, start_i - 3, start_i)):
                    if idx in int_matches:
                        if score_i < 0:
                            pos_int_cnt += 1
                        else:
                            neg_int_cnt += 1
                        break
            if score_i > 0.:
                pos_cnt += 1
                pos_so += score_i
            elif score_i < 0.:
                neg_cnt += 1
                neg_so += score_i
        # compute polar values (F_p, e_p, F_n, and e_n)
        A_p = float(pos_so) / (float(pos_cnt) or 1e10)
        F_p = min(A_p / np.log(P * pos_cnt + pos_int_cnt), 100.)
        e_p = min(A_p / (2. - np.log(P * pos_cnt)), 1.)
        A_n = float(neg_so) / (float(neg_cnt) or 1e10)
        F_n = max(A_n / np.log(P * neg_cnt + neg_int_cnt), -100.)
        e_n = max(A_n / (2. - np.log(P * neg_cnt)), -1.)
        self._logger.debug("F_p: %f; e_p: %f; F_n: %f; e_n: %f;",
                           F_p, e_p, F_n, e_n)
        if pos_cnt == 0:
            return self._final_sentiment(F_n, e_n)
        elif neg_cnt == 0:
            return self._final_sentiment(F_p, e_p)
        elif F_p - F_n > 0.1:
            return self._final_sentiment(F_p, e_p)
        elif F_n - F_p > 0.1:
            return self._final_sentiment(F_n, e_n)
        elif F_p + F_n > 0.:
            return self._final_sentiment(F_p, e_p)
        elif F_p + F_n < 0.:
            return self._final_sentiment(F_n, e_n)
        return 0.

    def _final_sentiment(self, F, e):
        """Compute final sentiment score from polarity and evidence.

        Args:
          F (float): polarity score
          e (float): evidence score

        Returns:
          float: final sentiment score

        """
        if abs(F) > 25 or abs(e) > 0.5:
            F /= 100.
        else:
            F = 0.
        self._logger.debug("final sentiment score: %f;", F)
        return F

    def _get_score(self, term, polarity, lexicon_score):
        # Jurek et al. use scores in the range [-100, 100]
        score = super(JurekAnalyzer, self).__init__(term, polarity,
                                                    lexicon_score)
        return score * 100.
