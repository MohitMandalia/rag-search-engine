from lib.search_utils import load_movies, load_stopwords, CACHE_PATH
import string
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
import os
import pickle
import math

stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set) # token: [doc_id1, doc_id2]
        self.docmap = {} # map document ID: document
        self.term_frequencies = defaultdict(Counter)

        self.index_path = CACHE_PATH/'index.pkl'
        self.docmap_path = CACHE_PATH/'docmap.pkl'
        self.term_frequencies_path = CACHE_PATH/'term_frequencies.pkl'

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_document(self, term):
        return sorted(list(self.index[term]))
    
    def get_tf(self, doc_id, term):
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Can only process 1 token")
        return self.term_frequencies[doc_id][token[0]]
    
    def get_idf(self, term):
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Can only process 1 token")
        token = token[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))


    def build(self):
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id,text)
            self.docmap[doc_id] = movie

    def save(self):
        # Create cache dir if not exist
        os.makedirs(CACHE_PATH, exist_ok=True)
        with open(self.index_path, 'wb') as file:
            pickle.dump(self.index, file)

        with open(self.docmap_path, 'wb') as file:
            pickle.dump(self.docmap, file)
        
        with open(self.term_frequencies_path, 'wb') as file:
            pickle.dump(self.term_frequencies, file)

    def load(self):
        with open(self.index_path, "rb") as file:
            self.index = pickle.load(file)
        with open(self.docmap_path, "rb") as file:
            self.docmap = pickle.load(file)
        with open(self.term_frequencies_path, 'rb') as file:
            self.term_frequencies = pickle.load(file)


def sanitize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("","",string.punctuation))
    return text

def tokenize_text(text):
    text = sanitize_text(text)
    stopwords = load_stopwords()
    res = []
    def _filter(tok):
        return tok and tok not in stopwords
    for tok in text.split():
        if _filter(tok):
            tok = stemmer.stem(tok)
            res.append(tok)
    return res

def has_matching_token(query_tokens, movie_tokens):
    for query_token in query_tokens:
        for movie_token in movie_tokens:
            if query_token in movie_token:
                return True
    return False

def search_command(query, n_results=5):
    movies = load_movies()
    idx = InvertedIndex()
    idx.load()
    seen, res = set(), []  
    query_tokens = tokenize_text(query)

    for qt in query_tokens:
        matching_doc_ids = idx.get_document(qt)
        for matching_doc_id in matching_doc_ids:
            if matching_doc_id in seen:
                continue
            seen.add(matching_doc_id)
            matching_doc = idx.docmap[matching_doc_id]
            res.append(matching_doc)

            if len(res) >= n_results:
                return res
    return res  


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()

def tf_command(doc_id, term):
    idx = InvertedIndex()
    idx.load()
    print(idx.get_tf(doc_id, term))

def idf_command(term):
    idx = InvertedIndex()
    idx.load()
    idf = idx.get_idf(term)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")
