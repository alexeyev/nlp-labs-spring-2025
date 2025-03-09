import math
import os
from collections import Counter

from tqdm import tqdm

"""
    Убедитесь, что использование PMI в чистом виде
      для извлечения коллокаций -- зачастую не лучший вариант.
    
    Подберите k так, чтобы в топе были не самые 
        редкие в романе сочетания (например, имя+отчество).
    
    Попробуйте придумать и применить полезные для извлечения устойчивых 
        сочетаний фильтры, например, по частям речи (?).
    
    На практике такие штуки можно было использовать 
        даже для извлечения именованных сущностей 
        из неменяющегося массива текстов.
"""


if __name__ == "__main__":

    if not os.path.exists("sentences.txt"):
        print("Please run `prepare_data.py` first. "
              "Make sure you have a stable Internet connection.")

    bigram_counter = Counter()
    unigram_counter = Counter()

    with open("sentences.txt", "r", encoding="utf-8") as rf:

        for line in tqdm(rf):
            tokens = line.strip().split(" ")

            for i in range(len(tokens) - 1):
                bigram_counter[(tokens[i], tokens[i + 1])] += 1
                unigram_counter[tokens[i]] += 1

            unigram_counter[tokens[-1]] += 1

    print("Common ngrams:")

    for (a, b), count in bigram_counter.most_common(6):
        print(f"[{a}]+[{b}]\t({count})")

    for w, count in unigram_counter.most_common(6):
        print(f"[{w}]\t({count})")

    print("Total unigrams (unique tokens):", len(unigram_counter))
    print("Total unigrams occurrences    :", sum(unigram_counter.values()))
    print("Total bigrams occurrences     :", sum(bigram_counter.values()))
    print()

    """
    Good collocation pairs have high PMI because the probability 
    of co-occurrence is only slightly lower than the probabilities 
    of occurrence of each word. Conversely, a pair of words whose 
    probabilities of occurrence are considerably higher than their 
    probability of co-occurrence gets a small PMI score. 
    """

    total_unigrams = sum(unigram_counter.values())
    total_bigrams = sum(bigram_counter.values())
    pmi_values = []

    # The additional factors of p (x,y) inside the logarithm are intended
    #   to correct the bias of PMI towards low-frequency events,
    #   by boosting the scores of frequent pairs.
    k = 4

    for (a, b), count in bigram_counter.items():
        # логарифмированный числитель: p(x, y)
        joint = math.log2(bigram_counter[(a, b)]) - math.log2(total_bigrams)
        joint = k * joint
        # логарифмированный знаменатель p(x)p(y) входит как -log(p(x)) - log(p(y))
        pmi_k = - math.log2(unigram_counter[a]) + 2 * math.log2(total_unigrams) \
                - math.log2(unigram_counter[b]) \
                + joint
        pmi_values.append((pmi_k, (a, b)))

    pmi_sorted = sorted(pmi_values, reverse=True)

    for pmi, (a, b) in pmi_sorted[:5]:
        print(f"[{a}]+[{b}]\t{pmi:.4f}\t"
              f"xy={bigram_counter[(a, b)]}\t"
              f"x={unigram_counter[a]}\t"
              f"y={unigram_counter[b]}")
