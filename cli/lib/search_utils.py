import json
from pathlib import Path

BM25_K1 = 1.5

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT/'data'
MOVIES_PATH = DATA_PATH/'movies.json'
STOP_WORDS_PATH = DATA_PATH/'stopwords.txt'
CACHE_PATH = PROJECT_ROOT/'cache'

def load_movies() -> list[dict]:
    with open(MOVIES_PATH, "r") as file:
        data = json.load(file)
    return data['movies']

def load_stopwords():
    with open(STOP_WORDS_PATH, "r") as file:
        data = file.read().splitlines()
    return data