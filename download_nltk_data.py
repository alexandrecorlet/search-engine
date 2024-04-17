import nltk


def download_nltk_data():
    """Download specific modules of NLTK"""
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('omw-1.4')


def main():
    download_nltk_data()


if __name__ == "__main__":
    main()

