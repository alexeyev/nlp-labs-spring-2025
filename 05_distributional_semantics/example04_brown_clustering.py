# pip install brown-clustering
import logging

from brown_clustering import BigramCorpus, BrownClustering

if __name__ == "__main__":
    import os

    if not os.path.exists("sentences.txt"):
        print("Please run `prepare_data.py` first. "
              "Make sure you have a stable Internet connection.")

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Logs to console
            logging.FileHandler('logs/brown.log', mode='a', encoding='utf-8')
        ]
    )

    with open("sentences.txt", "r", encoding="utf-8") as rf:
        sentences = [line.strip().split() for line in rf]

    # create a corpus
    corpus = BigramCorpus(sentences, alpha=0.5, min_count=5)

    # (optional) print corpus statistics:
    corpus.print_stats()

    # create a clustering
    clustering = BrownClustering(corpus, m=400)

    # train the clustering
    clusters = clustering.train()

    # use the clustered words
    for cl_id, cluster in enumerate(clusters):
        print(f"{cl_id}\t{cluster}")

    # Распечатайте коды и постройте по ним бинарное дерево,
    #   а потом придумайте, как сгруппировать кластеры,
    #   идя от листьев к корню

    # print(clustering.codes())
    # {'an': '110', 'another': '111', 'This': '00', 'example': '01', 'is': '10'}
