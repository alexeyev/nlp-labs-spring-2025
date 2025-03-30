import logging
import multiprocessing
import os.path

from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class SentenceIterator:
    def __init__(self, filename: str):
        self.filename = filename

    def __iter__(self):
        with open(self.filename, 'r', encoding='utf-8') as f:
            for line in f:
                yield line.strip().split()


def main():
    corpus_file = "tmp-train.txt"
    sentences = SentenceIterator(corpus_file)
    num_workers = max(multiprocessing.cpu_count() - 1, 1)
    model = Word2Vec(
        sentences=sentences,
        vector_size=50,
        window=5,
        min_count=10,
        sg=1,
        workers=num_workers,
        epochs=100
    )

    model.save("word2vec100e.bin")


if __name__ == '__main__':
    if os.path.exists("word2vec100e.bin"):
        emb_model = Word2Vec.load("word2vec100e.bin")
        print(emb_model.wv.most_similar(positive=["album"]))
        print(emb_model.wv.most_similar(positive=["song"]))
        print(emb_model.wv.most_similar(positive=["other"]))
        print(emb_model.wv.most_similar(positive=["aphex"]))
        print(emb_model.wv.most_similar(positive=["evil"]))
        print(emb_model.wv.most_similar(positive=["metal"]))
        print(emb_model.wv.most_similar(positive=["church"]))
    else:
        main()
