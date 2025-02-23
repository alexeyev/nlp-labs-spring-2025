""" Setting up the interface for search engines for Cranfield """


class StoredIndexEngine(object):
    def __init__(self, index_storage_path: str = "indexdir"):
        self.ix_path = index_storage_path

    def index(self, csv_file_path: str = "data/all.csv"):
        raise NotImplementedError

    def search(self, query: str, limit: int):
        raise NotImplementedError
