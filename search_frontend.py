from collections import Counter
from inverted_index_gcp import *
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.stem.porter import *

import nltk

nltk.download('stopwords')


class BM25_from_index:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5
    b : float, default 0.75
    index: inverted index
    """

    def __init__(self, index, DL, dir, k1=1.2, b=0.5):

        self.b = b
        self.k1 = k1
        self.index = index
        self.DL = DL
        self.N = len(DL)
        self.AVGDL = sum(DL.values()) / self.N
        self.dir = dir

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

    def search_merge(self, query, title_lst, N=12):

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
        simd = {}

        for doc_id, term in self.get_candidate_documents_and_scores(query).keys():
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
            simd[doc_id] = score
        temp = sorted([(doc_id, score) for doc_id, score in simd.items()], key=lambda x: x[1], reverse=True)[:N]

        ans = {}
        for doc_id, score in temp:
            ans[doc_id] = score * 0.5  # if body
        for doc_id, score in title_lst:
            if doc_id in ans.keys():
                ans[doc_id] += score * 0.50  # if body and title
            else:
                ans[doc_id] = score * 0.50
        return sorted([(doc_id, score) for doc_id, score in ans.items()], key=lambda x: x[1], reverse=True)[:12]

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


class MyFlaskApp(Flask):
    def __init__(
            self,
            import_name,
            static_url_path=None,
            static_folder="static",
            static_host=None,
            host_matching=False,
            subdomain_matching=False,
            template_folder="templates",
            instance_path=None,
            instance_relative_config=False,
            root_path=None,

    ):
        super().__init__(import_name, static_url_path, static_folder, static_host, host_matching, subdomain_matching,
                         template_folder, instance_path, instance_relative_config, root_path)
        self.inverted_body = InvertedIndex.read_index('body_index', 'body')

        objectRep = open("body_index/dict_dl", "rb")
        self.DL = dict(pickle.load(objectRep))
        objectRep.close()
        objectRep_pv = open('body_index/pageviews.pkl', 'rb')
        self.wid2pv = dict(pickle.load(objectRep_pv))
        objectRep_pv.close()

        objectRep_nfi = open("body_index/nfi", "rb")
        self.nfi = dict(pickle.load(objectRep_nfi))
        objectRep_nfi.close()

        objectRep_title = open("body_index/id_title.pickle", "rb")
        self.titles = dict(pickle.load(objectRep_title))
        objectRep_title.close()

        objectRep_title = open("title_index/titles_new.pickle", "rb")
        self.titles_new = dict(pickle.load(objectRep_title))
        objectRep_title.close()

        colnames = ['doc_id', 'pr']
        self.page_rank = pd.read_csv('anchor_index/pr_file.csv.gz', names=colnames, compression='gzip')
        self.inverted_title = InvertedIndex.read_index('title_index', 'title')
        self.inverted_anchor = InvertedIndex.read_index('anchor_index', 'anchor')

        self.BM25 = BM25_from_index(self.inverted_body, self.DL, "body_index")
        self.BM25_title = BM25_from_index(self.inverted_title, self.DL, "title_index")

    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

all_stopwords = english_stopwords.union(corpus_stopwords)


def tokenize(text, use_stemming=False):
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    if use_stemming:
        stemmer = PorterStemmer()
        for i in range(len(list_of_tokens)):
            list_of_tokens[i] = stemmer.stem(list_of_tokens[i])
    return list_of_tokens


import math


def merge_results(title_scores, body_scores, title_weight=0.2, text_weight=0.8, N=12):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).

    Parameters:
    -----------
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)

    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    dictionary of querires and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score).
    """
    ans = {}
    for doc_id, score in body_scores:
        ans[doc_id] = score * text_weight  # if body
    for doc_id, score in title_scores:
        if doc_id in app.titles.keys():
            if doc_id in ans.keys():
                ans[doc_id] += score * title_weight  # if body and title
            else:
                ans[doc_id] = score * title_weight  # if title and not body

    return sorted([(doc_id, score) for doc_id, score in ans.items()], key=lambda x: x[1], reverse=True)[:N]


def sim_body(index, query, dir):
    dict_query = Counter(query)
    len_query = len(query)
    q = 0
    sim = {}
    vec_size = {}
    for q_word in dict_query.keys():
        q += (dict_query[q_word] / len_query) ** 2
        pls = index.read_posting_list(q_word, dir)
        for doc_id, freq in pls:
            if (app.DL[doc_id] == 0 or index.df[q_word] == 0):
                print(q_word, doc_id, app.DL[doc_id], index.df[q_word])
            tf_idf_doc = (freq / app.DL[doc_id]) * math.log(len(app.DL) / index.df[q_word], 10)
            tf_idf_query = (dict_query[q_word] / len(query))
            if doc_id in sim.keys():
                sim[doc_id] += tf_idf_doc * tf_idf_query
            else:
                sim[doc_id] = tf_idf_doc * tf_idf_query
    for doc_id in sim.keys():
        sim[doc_id] = sim[doc_id] * (1 / (q ** 0.5)) * (1 / app.nfi[doc_id])
    temp = sorted([(doc_id, score) for doc_id, score in sim.items()], key=lambda x: x[1], reverse=True)[:12]
    return temp


def sim_title(index, query, dir):
    dict_query = Counter(query)
    len_query = len(query)
    q = 0
    nfi = {}
    sim = {}
    for q_word in dict_query.keys():
        q += (dict_query[q_word] / len_query) ** 2
        pls = index.read_posting_list(q_word, dir)
        for doc_id, freq in pls:
            if app.DL[doc_id] == 0 or index.df[q_word] == 0:
                continue
            else:
                if doc_id in nfi.keys():
                    nfi[doc_id] += (Counter(tokenize(app.titles[doc_id]))[q_word] / len(
                        tokenize(app.titles[doc_id]))) ** 2
                else:
                    nfi[doc_id] = (Counter(tokenize(app.titles[doc_id]))[q_word] / len(
                        tokenize(app.titles[doc_id]))) ** 2
                tf_idf_title = (freq / len(tokenize(app.titles[doc_id]))) * math.log(len(app.DL) / index.df[q_word], 10)
                tf_idf_query = (dict_query[q_word] / len(query))
                if doc_id in sim.keys():
                    sim[doc_id] += tf_idf_title * tf_idf_query
                else:
                    sim[doc_id] = tf_idf_title * tf_idf_query
    for doc_id in sim.keys():
        sim[doc_id] = sim[doc_id] * (1 / (q ** 0.5)) * (1 / nfi[doc_id] ** 0.5)
    temp = sorted([(doc_id, score) for doc_id, score in sim.items()], key=lambda x: x[1], reverse=True)[:12]
    return temp


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    q = tokenize(query)
    title = sim_title(app.inverted_title, q, "title_index")
    temp1 = app.BM25.search_merge(q, title)
    anchor = all_anchor_score_norm(q, app.inverted_anchor, len(q))
    temp = merge_results(anchor, temp1, 0.3, 0.7)
    for tup in temp:
        res.append((tup[0], app.titles[tup[0]]))
    return jsonify(res)


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    temp = sim_body(app.inverted_body, tokenize(query), "body_index")
    for tup in temp:
        res.append((tup[0], app.titles[tup[0]]))
    return jsonify(res)


def all_titles_score(query_to_search, index):
    candidates = {}
    for term in np.unique(query_to_search):
        if term in index.df.keys():
            list_of_doc = app.inverted_title.read_posting_list(term, "title_index")
            for doc_id, freq in list_of_doc:
                if doc_id in candidates.keys():
                    candidates[doc_id] += 1
                else:
                    candidates[doc_id] = 1

    return sorted([(doc_id, score) for doc_id, score in candidates.items()], key=lambda x: x[1], reverse=True)


def all_anchor_score(query_to_search, index):
    candidates = {}
    for term in np.unique(query_to_search):
        if term in index.df.keys():
            list_of_doc = app.inverted_anchor.read_posting_list(term, "anchor_index")
            for doc_id, freq in list_of_doc:
                if doc_id in candidates.keys():
                    candidates[doc_id] += 1
                else:
                    candidates[doc_id] = 1

    return sorted([(doc_id, score) for doc_id, score in candidates.items()], key=lambda x: x[1], reverse=True)


def all_anchor_score_norm(query_to_search, index, len_query):
    candidates = {}
    for term in np.unique(query_to_search):
        if term in index.df.keys():
            list_of_doc = app.inverted_anchor.read_posting_list(term, "anchor_index")
            for doc_id, freq in list_of_doc:
                if doc_id in candidates.keys():
                    candidates[doc_id] += 1
                else:
                    candidates[doc_id] = 1

    return sorted([(doc_id, score / len_query) for doc_id, score in candidates.items()], key=lambda x: x[1],
                  reverse=True)


@app.route("/search_title")
def search_title():
    """ Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    """
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    temp = all_titles_score(tokenize(query), app.inverted_title)
    for tup in temp:
        res.append((tup[0], app.titles[tup[0]]))

    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    dic = all_anchor_score(tokenize(query), app.inverted_anchor)

    for tup in dic:
        if tup[0] in app.titles.keys():
            res.append((tup[0], app.titles_new[tup[0]]))
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''

    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for id in wiki_ids:
        if id not in app.DL.keys():
            res.append(0)
        else:
            res.append(app.page_rank.loc[app.page_rank["doc_id"] == id, 'pr'].values[0])
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for id in wiki_ids:
        if id in app.DL.keys():
            res.append(app.wid2pv[id])
        else:
            res.append(0)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
