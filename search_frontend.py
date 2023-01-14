from collections import Counter
from inverted_index_gcp import *
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
# import gensim
from nltk.stem.porter import *
from BM25fromInvertedIndex import BM25_from_index

import nltk

# from nltk.stem.porter import *
# from nltk.corpus import stopwords
nltk.download('stopwords')


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

        colnames = ['doc_id', 'pr']
        self.page_rank = pd.read_csv('anchor_index/pr_file.csv.gz', names=colnames, compression='gzip')
        self.inverted_title = InvertedIndex.read_index('title_index', 'title')
        self.inverted_anchor = InvertedIndex.read_index('anchor_index', 'anchor')



        self.inverted_body_stem = InvertedIndex.read_index('body_stem_index', 'body_stem')



        self.BM25 = BM25_from_index(self.inverted_body, self.DL, "body_index")
        self.BM25_stem = BM25_from_index(self.inverted_body_stem, self.DL, "body_stem_index")
        print("===================================================")

    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
print("===================================================")

app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
print("===================================================")


print("===================================================")


print("===================================================")

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

all_stopwords = english_stopwords.union(corpus_stopwords)


def get_top_n(sim_dict, N=20):
    return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]


def tokenize(text, use_stemming=False):
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    if use_stemming:
        stemmer = PorterStemmer()
        for i in range(len(list_of_tokens)):
            list_of_tokens[i] = stemmer.stem(list_of_tokens[i])
    return list_of_tokens


import math


def generate_query_tfidf_vector(query_to_search, index):
    epsilon = .0000001
    total_vocab_size = len(index.term_total)
    Q = np.zeros((total_vocab_size))
    term_vector = list(index.term_total.keys())
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.term_total.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log((len(app.DL)) / (df + epsilon), 10)  # smoothing

            try:
                ind = term_vector.index(token)
                Q[ind] = tf * idf
            except:
                pass
    return Q


def get_posting_iter(index):
    words, pls = zip(*index.posting_lists_iter())
    return words, pls


def cosine_similarity(D, Q):
    # YOUR CODE HERE
    from numpy import linalg as LA
    ans = {}
    for index, row in D.iterrows():
        value = Q.dot(row) / (LA.norm(Q) * LA.norm(row))
        ans[index] = value
    return ans


def merge_results(title_scores, body_scores, title_weight=0.2, text_weight=0.8, N=20):
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
    # YOUR CODE HERE
    ans = {}
    for doc_id, score in body_scores:
        ans[doc_id] = score * text_weight  # if body
    for doc_id, score in title_scores:
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
            tf_idf_doc = (freq / app.DL[doc_id]) * math.log(len(app.DL) / index.df[q_word], 10)
            tf_idf_query = (dict_query[q_word] / len(query))
            if doc_id in sim.keys():
                sim[doc_id] += tf_idf_doc * tf_idf_query
            else:
                sim[doc_id] = tf_idf_doc * tf_idf_query
    for doc_id in sim.keys():
        sim[doc_id] = sim[doc_id] * (1 / (q ** 0.5)) * ((1 / app.nfi[str(doc_id)]) ** 0.5)
    temp = sorted([(doc_id, score) for doc_id, score in sim.items()], key=lambda x: x[1], reverse=True)[:20]
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
    # BM25 body
    # res = []
    # query = request.args.get('query', '')
    # if len(query) == 0:
    #     return jsonify(res)
    # q = tokenize(query)
    # temp = app.BM25.search(q)
    # for tup in temp:
    #     res.append((tup[0], app.titles[tup[0]]))
    # return jsonify(res)

    # basic search
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    body = sim_body(app.inverted_body,tokenize(query),"body_index")
    title = all_titles_score(tokenize(query), app.inverted_title)
    merged_list = merge_results(title, body)
    for tup in merged_list:
        res.append((tup[0], app.titles[tup[0]]))
    return jsonify(res)


#     search with stemming
#     res = []
#     query = request.args.get('query', '')
#     if len(query) == 0:
#         return jsonify(res)
#     stemQ = tokenize(query,use_stemming=True)
#     body = sim_body(app.inverted_body_stem,stemQ,'body_stem')
#     title = all_titles_score(stemQ, app.inverted_title)
#     merged_list = merge_results(title, body)
#     for tup in merged_list:
#         res.append((tup[0], app.titles(tup[0])))
#     return jsonify(res)

# search based on title , body_stem and anchor
#     res = []
#     query = request.args.get('query', '')
#     if len(query) == 0:
#         return jsonify(res)
#     stemQ = tokenize(query,use_stemming=True)
#     body = sim_body(app.inverted_body_stem,stemQ,'body_stem')
#     title = all_titles_score(stemQ, app.inverted_title)
#     anchor = all_anchor_score(stemQ,app.inverted_anchor)
#     merged_list_body_title = merge_results(title, body)
#     merged_list = merge_results(anchor , merged_list_body_title , 0.1,0.9)
#     for tup in merged_list:
#         res.append((tup[0], app.titles(tup[0])))
#     return jsonify(res)


# BM25 body stem
#     res = []
#     query = request.args.get('query', '')
#     if len(query) == 0:
#         return jsonify(res)
#     q = tokenize(query)
#     temp = app.BM25_stem.search(q)
#     for tup in temp:
#         res.append((tup[0], app.titles[tup[0]]))
#     return jsonify(res)


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

    # res = []
    # query = request.args.get('query', '')
    # if len(query) == 0:
    #     return jsonify(res)
    #
    # toke_query = tokenize(query)
    # dict_query = Counter(toke_query)
    # sim = {}
    # vec_size = {}
    # for q_word in dict_query.keys():
    #     pls = app.inverted_body.read_posting_list(q_word, "body_index")
    #     v_size = 0
    #     for doc_id, tf_score in pls:
    #         if doc_id in sim.keys():
    #             sim[doc_id] += dict_query[q_word] * tf_score
    #             v_size = tf_score ** 2
    #         else:
    #             sim[doc_id] = dict_query[q_word] * tf_score
    # for doc_id in sim.keys():
    #     sim[doc_id] = sim[doc_id] * (1 / len(toke_query)) * ((1 / app.DL[doc_id]) ** 0.5)
    # temp = sorted([(doc_id, score) for doc_id, score in sim.items()], key=lambda x: x[1], reverse=True)[:20]
    # for tup in temp:
    #     res.append((tup[0], app.titles[tup[0]]))
    # return jsonify(res)

    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    toke_query = tokenize(query)
    dict_query = Counter(toke_query)
    len_query = len(toke_query)
    q = 0
    sim = {}
    vec_size = {}
    for q_word in dict_query.keys():
        q += (dict_query[q_word] / len_query) ** 2
        pls = app.inverted_body.read_posting_list(q_word, "body_index")
        for doc_id, freq in pls:
            tf_idf_doc = (freq / app.DL[doc_id]) * math.log(len(app.DL) / app.inverted_body.df[q_word], 10)
            # tf_idf_query = (dict_query[q_word] / len(toke_query)) * math.log(len(app.DL) / app.inverted_body.df[q_word], 10)
            tf_idf_query = (dict_query[q_word] / len(toke_query))
            if doc_id in sim.keys():
                sim[doc_id] += tf_idf_doc * tf_idf_query
            else:
                sim[doc_id] = tf_idf_doc * tf_idf_query
    for doc_id in sim.keys():
        sim[doc_id] = sim[doc_id] * (1 / (q ** 0.5)) * ((1 / app.nfi[doc_id]) ** 0.5)
    temp = sorted([(doc_id, score) for doc_id, score in sim.items()], key=lambda x: x[1], reverse=True)[:20]
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
            list_of_doc = app.inverted_title.read_posting_list(term, "anchor_index")
            for doc_id, freq in list_of_doc:
                if doc_id in candidates.keys():
                    candidates[doc_id] += 1
                else:
                    candidates[doc_id] = 1

    return candidates


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

    # END SOLUTION
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
    # BEGIN SOLUTION
    dic = all_anchor_score(tokenize(query), app.inverted_anchor)
    temp = sorted([(doc_id, score) for doc_id, score in dic.items()], key=lambda x: x[1], reverse=True)
    for tup in temp:
        res.append((tup[0], app.titles[tup[0]]))
    # END SOLUTION
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
