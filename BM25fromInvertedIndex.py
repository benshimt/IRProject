import math
from collections import Counter
from itertools import chain

import time

# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
import numpy as np


class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5
    b : float, default 0.75
    index: inverted index
    """

    def __init__(self, index, DL, dir, k1=1.5, b=0.75):

        self.b = b
        self.k1 = k1
        self.index = index
        self.DL = DL
        self.N = len(DL)
        self.AVGDL = sum(DL.values()) / self.N
        self.dir = dir

        # self.pls = zip(*self.index.posting_lists_iter())

    def calc_idf(self, query):

        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        Returns:
        -----------
        idf: dictionary of idf scores. As follows:

                                                    key: term

                                                    value: bm25 idf score
        """

        idf = {}
        for term in query:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + ((self.N - n_ti + 0.5) / (n_ti + 0.5)))
            else:
                pass
        return idf

    def get_candidate_documents_and_scores(self, query_to_search):
        candidates = {}
        for term in np.unique(query_to_search):
            if term in self.index.df.keys():
                list_of_doc = self.index.read_posting_list(term, self.dir)
                normlized_tfidf = [(doc_id, (freq / self.DL[doc_id]) * math.log(len(self.DL) / self.index.df[term], 10))
                                   for
                                   doc_id, freq in list_of_doc]

                for doc_id, tfidf in normlized_tfidf:
                    candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

        return candidates

    def search(self, query, N=20):

        """

        This function calculate the bm25 score for given query and document.

        We need to check only documents which are 'candidates' for a given query.

        This function return a dictionary of scores as the following:

                                                                    key: query_id

                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.



        Parameters:

        -----------

        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        doc_id: integer, document id.



        Returns:

        -----------

        score: float, bm25 score.

        """
        sim = {}

        for doc_id, term in self.get_candidate_documents_and_scores(query).keys():
            sim[doc_id] = self._score(query, doc_id)
        temp_body = sorted([(doc_id, score) for doc_id, score in sim.items()], key=lambda x: x[1], reverse=True)[:N]

        sim2 = {}

        for doc_id, term in self.get_candidate_titles_and_scores(query).keys():
            sim[doc_id] = self._score(query, doc_id)
        temp = sorted([(doc_id, score) for doc_id, score in sim.items()], key=lambda x: x[1], reverse=True)[:N]

        return temp

    def search_merge(self, query, title_lst, N=20):

        """

        This function calculate the bm25 score for given query and document.

        We need to check only documents which are 'candidates' for a given query.

        This function return a dictionary of scores as the following:

                                                                    key: query_id

                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.



        Parameters:

        -----------

        query: list of token representing the query. For example: ['look', 'blue', 'sky']

        doc_id: integer, document id.



        Returns:

        -----------

        score: float, bm25 score.

        """
        sim = {}

        for doc_id, term in self.get_candidate_documents_and_scores(query).keys():
            sim[doc_id] = self._score(query, doc_id)
        temp = sorted([(doc_id, score) for doc_id, score in sim.items()], key=lambda x: x[1], reverse=True)[:N]

        ans = {}
        for doc_id, score in temp:
            ans[doc_id] = score * 0.8  # if body
        for doc_id, score in title_lst:
            if doc_id in ans.keys():
                ans[doc_id] += score * 0.2  # if body and title
            else:
                ans[doc_id] = score * 0.2
        return sorted([(doc_id, score) for doc_id, score in ans.items()], key=lambda x: x[1], reverse=True)[:20]

    def _score(self, query, doc_id):

        """
        This function calculate the bm25 score for given query and document.
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.
        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0
        doc_len = self.DL[doc_id]
        self.idf = self.calc_idf(query)
        for term in query:
            if term in self.index.term_total.keys():
                term_frequencies = dict(self.index.read_posting_list(term, self.dir))
                if doc_id in term_frequencies.keys():
                    freq = term_frequencies[doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score

    def get_candidate_titles_and_scores(self, query_to_search, index):
        candidates = {}
        for term in np.unique(query_to_search):
            if term in self.index.df.keys():
                list_of_doc = index.read_posting_list(term, self.dir)
                normlized_tfidf = [(doc_id, (freq / self.DL[doc_id]) * math.log(len(self.DL) / self.index.df[term], 10))
                                   for
                                   doc_id, freq in list_of_doc]

                for doc_id, tfidf in normlized_tfidf:
                    candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

        return candidates
