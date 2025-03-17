import logging
from collections import Counter
from itertools import chain
from typing import Iterable, Tuple, Dict

import numpy as np
from scipy.sparse import dok_matrix, csr_matrix, spmatrix
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm


# does not work with sparse matrices
# from scipy.spatial.distance import cdist


def make_window_iter(index: int, half_window_size: int, length: int):
    left_part = range(max(0, index - half_window_size),
                      max(0, index - 1))
    right_part = range(min(index + 1, length - 1),
                       min(length - 1, index + half_window_size))
    return chain(left_part, right_part)


def build_word_context_counters(lines: Iterable[str]) -> Tuple[Counter, Counter]:
    # unigram counter
    vocab = Counter()

    # word-context tuples counter
    wc_count = Counter()

    for line in tqdm(lines, "reading data, filling dicts"):
        sentence = [t for t in line.strip().split()]
        vocab.update(sentence)
        for idx, target_word in enumerate(sentence):
            window = make_window_iter(idx, window_dist, len(sentence))
            for ctx_idx in window:
                wc_count[(target_word, sentence[ctx_idx])] += 1

    return vocab, wc_count


def find_top_k(wc_like_matrix: spmatrix,
               word_id: int,
               id2tok: Dict[int, str],
               k: int = 5):
    x, y = wc_like_matrix[word_id], wc_like_matrix

    # Could use vector indexing
    dists = pairwise_distances(X=x, Y=y, metric="cosine")[0]
    ordered_by_dist = np.argsort(dists)

    for neighbour in ordered_by_dist[1:k + 1]:
        # print(f"> {id2tok[neighbour]} ({dists[neighbour]:.3f})")
        yield id2tok[neighbour], dists[neighbour]


if __name__ == "__main__":
    import os

    """
        Убедитесь, что на голых счётчиках далеко не уедешь
        
        Попробуйте использовать PMI-матрицу (её можно построить эффективно
          с помощью матричных операций, не надо писать больших циклов)
        
        Воспользуйтесь подходами, рассказанными на лекции -- 
          должно стать лучше, но аккуратнее с возведением в степень,
          после неё нужно быть очень осторожным с обрезкой по нулю
        
        А может, если сделать матричное разложение, станет ещё лучше?
        
        Дерзайте.
    """

    if not os.path.exists("sentences.txt"):
        print("Please run `prepare_data.py` first. "
              "Make sure you have a stable Internet connection.")

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Logs to console
            logging.FileHandler('logs/count-based.log', mode='a', encoding='utf-8')
        ]
    )

    sentences, window_dist = [], 2

    with open("sentences-larger.txt", "r", encoding="utf-8") as rf:
        unigram_counter, wc_counter = build_word_context_counters(rf)

    total_tok = sum(unigram_counter.values())
    total_tok_uniq = len(unigram_counter.keys())
    print(f"Unique: {total_tok_uniq}, total: {total_tok}")

    tok2id = {k: i for i, (k, _) in enumerate(unigram_counter.most_common())}
    id2tok = {i: word for word, i in tok2id.items()}
    unigram_array = np.zeros((len(tok2id),))

    for tok, i in tok2id.items():
        unigram_array[i] = unigram_counter.get(tok, 0)

    wc_count_matrix = dok_matrix((len(tok2id), len(tok2id)))

    for (tgt, ctx), count in tqdm(wc_counter.items(), "building WC matrix"):
        wc_count_matrix[tok2id[tgt], tok2id[ctx]] = count

    wc_count_matrix = csr_matrix(wc_count_matrix)
    pmi_matrix = wc_count_matrix.copy()
    """
        pmi(x,y) = log p(x,y)/p(x)/p(y)
    """

    seed_words = ["красный", "яблоко", "пушкин"]

    for query in seed_words:
        print(query)
        for word, s in find_top_k(wc_count_matrix, tok2id[query.lower()], id2tok, k=5):
            print(f" > {word} ({s:.3f})")
