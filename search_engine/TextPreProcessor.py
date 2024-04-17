from nltk.corpus import stopwords
from search_engine.LemmatizerWithPOSTagger import LemmatizerWithPOSTagger


class TextPreProcessor():

    def __init__(self):
        self.lemmatizer = LemmatizerWithPOSTagger()

    def _remove_punctuation(self, text):
        preprocessed_text = [word for word in text if word.isalnum()]
        return preprocessed_text

    def _remove_stopwords(self, text):
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in text if word.lower() not in stop_words]
        return filtered_words

    def preprocess_text(self, text):
        preprocessed_text = self.lemmatizer.pos_tag_lemma(text)
        preprocessed_text = self._remove_punctuation(preprocessed_text)
        preprocessed_text = self._remove_stopwords(preprocessed_text)
        preprocessed_text = [word.lower() for word in preprocessed_text]
        return preprocessed_text

