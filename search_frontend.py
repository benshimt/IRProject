from collections import Counter
from inverted_index_gcp import *
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
import gensim


# import nltk
# from nltk.stem.porter import *
# from nltk.corpus import stopwords
# nltk.download('stopwords')


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
        self.inverted_body = None
        self.DL = None
        self.titles = None
        self.page_rank = None
        self.inverted_title = None
        self.inverted_anchor = None
        self.wid2pv = None
        self.inverted_body_stem = None
        self.nfi = None


    def run(self, host=None, port=None, debug=None, **options):
        # self.wid2pv = dict(pickle.load(open('body_index/pageviews.pkl', 'rb')))
        self.inverted_body = InvertedIndex.read_index('body_index', 'body')
        objectRep = open("body_index/dict_dl", "rb")
        self.DL = dict(pickle.load(objectRep))
        objectRep_nfi = open("body_index/nfi", "rb")
        self.nfi = dict(pickle.load(objectRep_nfi))
        objectRep_title = open("body_index/id_title.pickle", "rb")
        self.titles = dict(pickle.load(objectRep_title))
        colnames = ['doc_id', 'pr']
        self.page_rank = pd.read_csv('anchor_index/pr_file.csv.gz', names=colnames, compression='gzip')
        self.inverted_title = InvertedIndex.read_index('title_index', 'title')
        self.inverted_anchor = InvertedIndex.read_index('anchor_index', 'anchor')
        self.inverted_body_stem = InvertedIndex.read_index('body_stem_index', 'body_stem')
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


def get_candidate_documents_and_scores(query_to_search, index, words, pls):
    candidates = {}
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            normlized_tfidf = [(doc_id, (freq / app.DL[str(doc_id)]) * math.log(len(app.DL) / index.df[term], 10)) for
                               doc_id, freq in list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def generate_document_tfidf_matrix(query_to_search, index, words, pls):
    total_vocab_size = len(index.term_total)
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words,
                                                           pls)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = index.term_total.keys()

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf

    return D


def get_top_n(sim_dict, N=3):
    return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]


def tokenize(text):
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
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


def get_topN_score_for_queries(queries_to_search, index, N=20):
    words, pls = get_posting_iter(index)
    D = generate_document_tfidf_matrix(queries_to_search, index, words, pls)
    Q = generate_query_tfidf_vector(queries_to_search, index)
    tup = get_top_n(cosine_similarity(D, Q), N)
    return tup




@app.route("/search?")
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
    # BEGIN SOLUTION

    # END SOLUTION
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
        q += (dict_query[q_word]/len_query) ** 2
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
        sim[doc_id] = sim[doc_id] * (1 / (q ** 0.5)) * ((1 / app.nfi[str(doc_id)]) ** 0.5)
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
            for doc_id,freq in list_of_doc:
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
            res.append(app.wid2pv[str(id)])
        else:
            res.append(0)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
