import csv
from Document import Document
from InvertedIndex import InvertedIndex

class InMemorySearchEngine():
    
    def __init__(self, path_to_corpus):
        self.corpus = self._load_corpus(path_to_corpus)
        self.inverted_index = InvertedIndex(self.corpus)

    def _load_corpus(self, path_to_corpus):
        with open(path_to_corpus, 'r', encoding='utf-8') as csv_file:
            corpus = [Document(doc_id, text) for doc_id, text in csv.reader(csv_file)]
            return corpus

    def search(self, query):
        return self.inverted_index.get_top_k(query, k=5)

