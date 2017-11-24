#!/usr/bin/env python
# -*- mode: python; coding: utf-8; -*-

##################################################################
# Documentation

##################################################################
# Imports
from __future__ import absolute_import, unicode_literals, print_function

from six.moves import range

from cgsa.constants import COND_PROB_FILE
from cgsa.lexicon.base import CondProbLexiconBaseAnalyzer
from itertools import chain
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

##################################################################
# Variables and Constants
PARAM_GRID = {"n_neighbors": [3, 5, 7],
              "weights": ["uniform", "distance"]}

##################################################################
# Classes
class KolchynaAnalyzer(CondProbLexiconBaseAnalyzer):
    """Lexicon-based sentiment analysis method of Jurek et al.

    """

    def __init__(self, lexicons=[], cond_probs=COND_PROB_FILE):
        """Attributes:


        Notes:
          Automatic Lexicon Generation.

          In this study we aimed to create a lexicon specifi- cally oriented
          for sentiment analysis of Twitter messages. For this purpose we used
          the approach described in 4: ``Constructing a lexicon from trained
          data'' and the training dataset from Mark Hall (Hall, 2012) that is
          comprised of manually labelled 41403 pos itive Twitter messages and
          8552 negative Twitter messages. The method to generate a sentiment
          lexicon was implemented as follows:

            1. Pre-processing of the dataset: POS tags were assigned to all
              words in the dataset; words were lowered in case; BOW was created
              by tokenising the sentences in the dataset.

            2. The number of occurrences of each word in positive and negative
              sentences from the training dataset was calculated.

            3. The positive polarity of each word was calculated by dividing
              the number of occurrences in positive sentences by the number of
              all occurrences:

          Based on the positive score of the word we can make a decision about
          its polarity: the word is considered positive, if its positive score
          is above 0.6; the word is considered neutral, if its positive score
          is in the range [0.4; 0.6]; the word is considered negative, if the
          positive score is below 0.4. Since the positive score of the word
          ``pleasant'' is 0.73, it is considered to carry positive
          sentiment. Sentiment scores of some other words from the experiment
          are presented in Table 4.

          In our study all words from the Bag-of-Words with a polarity in the
          range [0.4; 0.6] were removed, since they do not help to classify the
          text as positive or negative.  The sentiment scores of the words were
          mapped into the range [-1;1] by using the following formula: P

            PolarityScore = 2 * positiveSentScore - 1.

          In this study we manually constructed a lexicon of emoticons,
          abbreviations and slang words commonly used in social-media to
          express emotions (EMO). Example of tokens from our lexicon are
          presented in Table 5. We aimed to analyse how performance of the
          classic opinion lexicon (OL) (Hu and Liu, 2004) can be improved by
          enhancing it with our EMO lexicon. We also expanded the lexicon
          further by incorporating words from the automatically created lexicon
          (AUTO). The process of automatic lexicon creation was described in
          detail in the previous section.

          To compare the performance of three lexicon combinations we need to
          assign positive, negative or neutral labels to the tweets based on
          the calculated sentiment scores, and compare the predicted labels
          against the true labels of tweets. For this purpose we employ a
          k-means clustering algorithm, using Simple Average and Logarithmic
          scores as features.

        Negation Handling.

          We implemented negation handling using simple, but effec- tive
          strategy: if negation word was found, the sentiment score of every
          word appear- ing between a negation and a clause-level punctuation
          mark (.,!?:;) was reversed (Pang et al., 2002). There are, however,
          some grammatical constructions in which a negation term does not have
          a scope. Some of these situations we implemented as exceptions:

          Exception Situation 1: Whenever a negation term is a part of a phrase
          that does not carry negation sense, we consider that the scope for
          negation is absent and the polarity of words is not
          reversed. Examples of these special phr

          Exception Situation 2: A negation term does not have a scope when it
          occurs in a negative rhetorical question. A negative rhetorical
          question is identified by the following heuristic. (1) It is a
          question; and (2) it has a negation term within the first three words
          of the question. For example:

          ``Did not I enjoy it?''
          ``Wouldn't you like going to the cinema?

        """
        self._cond_probs = self._load_cond_probs(cond_probs)
        super(KolchynaAnalyzer, self).__init__(lexicons)
        self.name = "kolchyna"
        self._clf = GridSearchCV(
            KNeighborsClassifier(),
            param_grid=PARAM_GRID,
            scoring="f1_macro")

    def train(self, train_x, train_y, dev_x, dev_y,
              **kwargs):
        # no training is required for this method
        scores = [self._compute_so(tweet_i)
                  for tweet_i in chain(train_x, dev_x)]
        self._logger.debug("training scores: %r", scores)
        labels = [label_i
                  for label_i in chain(train_y, dev_y)]
        self._logger.debug("training labels: %r", labels)
        # train k-NN regressor
        self._clf.fit(scores, labels)
        # check thresholds for every score type
        self._logger.info("best score: %f; best params: %r;",
                          self._clf.best_score_, self._clf.best_params_)

    def _compute_so(self, tweet):
        forms = [self._preprocess(w_i.form)
                 for w_i in tweet]
        lemmas = [self._preprocess(w_i.lemma)
                  for w_i in tweet]
        tags = [w_i.tag for w_i in tweet]
        match_input = [(f, l, t)
                       for f, l, t
                       in zip(forms, lemmas, tags)]
        # match blocking constructs
        boundaries = self._find_boundaries(match_input)
        self._logger.debug("boundaries: %r", boundaries)
        # match negations
        negated_words = set()
        for _, start, end in self._negations.search(match_input):
            # skip exceptions
            if ((end < len(lemmas) - 1
                 and ((lemmas[end] == "kein" and lemmas[end + 1] == "fall")
                      or (lemmas[end] == "nicht"
                          and lemmas[end + 1] == "nur")))
                    or '?' in self._get_sent_punct(end, forms, boundaries)):
                continue
            negated_words |= set(range(start,
                                       self._find_next_boundary(end,
                                                                boundaries,
                                                                left=False)))
        self._logger.debug("negated words: %r", negated_words)
        # match polar terms
        polterm_matches = self._join_scores(
            self._polar_terms.search(match_input))
        self._logger.debug("polterm_matches: %r;", polterm_matches)
        pol_score, pol_cnt = 0., 0
        for score_i, start_i, end_i in polterm_matches:
            if set(range(start_i, end_i + 1)) & negated_words:
                score_i *= -1.
            pol_score += score_i
            pol_cnt += 1
        mean_score = log_score = 0.
        if pol_cnt != 0:
            mean_score = pol_score / float(pol_cnt)
            if abs(mean_score) > 0.1:
                log_score = np.log10(abs(10. * mean_score))
                if mean_score < 0:
                    log_score *= -1
        self._logger.debug("mean_score: %f; log_score: %f;",
                           mean_score, log_score)
        return (mean_score, log_score)

    def _get_term_score(self, term, polarity, lexicon_score):
        """Obtain term's score from conditional probabilities table.

        Args:
          term (str): term whose conditional probability should be checked
          polarity (str): polarity class of the term
          lexicon_score (float): sentiment score of the term from the lexicon
            (used as fallback)

        Returns:
          float: conditional probability of the term or its original
            lexicon score

        """
        if term in self._cond_probs:
            chck_term = term
        else:
            chck_term = term.lower()
            if chck_term not in self._cond_probs:
                return lexicon_score

        pos_prob, _ = self._cond_probs[chck_term]
        if 0.4 < pos_prob < 0.6:
            return 0.
        return 2 * pos_prob - 1

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
        mean_score, log_score = self._compute_so(msg)
        yvec[:] = self._clf.predict_proba([[mean_score, log_score]])
        self._logger.debug("resulting yvec: %r;", yvec)
