# IRProject

## code Structure:
Inverted Index class inside the file Inverted_index_gcp.py imported to the main file of the program search_frontend.py


### inside search_frontend.py:
1. BM25 class : implementation of BM25 score formula and retrive the top 10 documents.
2. the Flask APP includes implementation of cosine similarity to the title and the body of the documents , implementation of each search function .
the run method of the app resposible for eading and creating all the necessary files and objects

### Search function
#### main search
retrive documents while the body is scored by BM25 fomula , the titles are scored by cosine similarity and the anchor is scored binary
#### search_body
retrive the top 10 documents based on body scored by cosine similarity
#### search_title
retrive all the documents based on title scored binary
#### search_anchor
retrive all the documents based on anchor scored binary
#### pageRank
calculate the page rank for each wiki file and return it as a list
#### pageRank
calculate the page view for each wiki file and return it as a list
