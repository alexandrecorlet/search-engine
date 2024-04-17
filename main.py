from search_engine.InMemorySearchEngine import InMemorySearchEngine


def main():
    path_to_corpus = "/Users/alexandrecorlet/search-engine/sample_data/docs.csv"
    searchEngine = InMemorySearchEngine(path_to_corpus)
    query = "masashi kishimoto and ninjas"
    print(searchEngine.search(query))


if __name__ == "__main__":
    main()

