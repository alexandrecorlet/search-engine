import math
from collections import defaultdict
from search_engine.TextPreProcessor import TextPreProcessor
from search_engine.Document import Document


class InvertedIndex():
    
    def __init__(self, corpus=None):
        self.index = defaultdict(dict)
        self.idf = dict()
        self.doc_lengths = dict()
        self.avg_doc_length = 0.0
        self.text_preprocessor = TextPreProcessor()
        if corpus:
            self._build(corpus)


    def _build(self, corpus):
        """Build the Inverted Index, compute the TF,
        average document length, and IDF"""

        for doc in corpus:
            doc_id = doc.doc_id
            doc_text = doc.text
            # Store the length of the i-th doc to compute the Okapi BM25 later on
            self.doc_lengths[doc_id] = len(doc_text)
            # Increment the average document length 
            self.avg_doc_length += len(doc_text) 
            # Preprocess the document text
            preprocessed_text = self.text_preprocessor.preprocess_text(doc_text)
            for token in preprocessed_text:
                # Add document to token list and update TF
                self.index[token][doc_id] = self.index[token].get(doc_id, 0) + 1

        # Compute average document length
        self.avg_doc_length /= len(corpus)
        
        # Compute IDF of each token 
        for token, docs in self.index.items():
            df = len(self.index[token])
            idf = math.log(len(corpus) / (df + 1)) + 1
            self.idf[token] = idf

    def get_top_k(self, query, top_k=5, k1=1.5, b=0.75):
        """Retrieve top k elements."""
        # Preprocess the query
        preprocessed_query = self.text_preprocessor.preprocess_text(query)
        # Initialize dict to store the score of each document 
        scores = defaultdict(float)
        # Iterate through each token of the query to compute the score
        for token in preprocessed_query:
            if token in self.index:
                # Apply the Okapi BM25 algorithm to compute/update the score of the i-th document
                idf = self.idf[token]
                for doc_id, tf in self.index[token].items():
                    num = tf * (k1 + 1)
                    den = tf + k1 * ((1 - b + b) / (self.doc_lengths[doc_id] / self.avg_doc_length))
                    scores[doc_id] += idf * num / den

        # Sort documents in decreasing order based on their score. 
        sorted_scores = sorted(scores.items(), key=lambda x:x[1], reverse=True)
        # Extract top k doc from sorted scores
        top_k_docs = [doc_id for doc_id, _ in sorted_scores[:top_k]]
        # Return top K docs :D
        return top_k_docs

