import nltk
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet

from search_engine.Splitter import Splitter


class LemmatizerWithPOSTagger(object):

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.splitter = Splitter()

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('A'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # Default POS tagger is NOUN
            return wordnet.NOUN

    def pos_tag_lemma(self, text):
        # Split the text into tokens
        tokens = self.splitter.split(text)

        # Flatten the list of tokens
        tokens = [word for sentence in tokens for word in sentence]

        # Find the pos tag for each token
        pos_tokens = nltk.pos_tag(tokens)

        # Lemmatization using pos tag
        lemmas = []
        for word, pos_tag in pos_tokens:
            lemmas.append(self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos_tag)))

        return lemmas

