import logging
import sys
from difflib import SequenceMatcher

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # в стандартный вывод
        logging.StreamHandler(),
        # в файл
        logging.FileHandler('strings-clustering.log', mode='a', encoding='utf-8')
    ]
)


def longest_common_substring(s1, s2):
    """ Наибольшая общая подстрока """

    m, n = len(s1), len(s2)
    # Матрица для дин. прогр
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0

    for i in range(m):
        for j in range(n):
            if s1[i] == s2[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
                if dp[i + 1][j + 1] > max_length:
                    max_length = dp[i + 1][j + 1]
            else:
                dp[i + 1][j + 1] = 0

    return max_length


def normalized_lcs_distance(s1, s2):
    """
        Нормализация длины общей подстроки как мера сходства
        similarity = LCS_length(s1, s2) / max(len(s1), len(s2))
        тогда расстояние
        distance = 1 - similarity.
    """
    if not s1 and not s2:
        return 0.0
    max_len = max(len(s1), len(s2))
    lcs_len = longest_common_substring(s1, s2)
    similarity = lcs_len / max_len
    return 1.0 - similarity


def difflib_distance(s1, s2):
    """ Расстояние на основе сходства Gestalt """
    return 1.0 - SequenceMatcher(a=s1, b=s2).ratio()


def compute_condensed_distance_matrix(strings):
    """ Верхняя диагональная матрица со всеми расстояниями """
    n = len(strings)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = normalized_lcs_distance(strings[i], strings[j])
            # dist = difflib_distance(strings[i], strings[j])
            dists.append(dist)
    return np.array(dists)


def read_strings_from_file(filepath):
    """ Чтение строк из файла """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def cluster_strings(strings, max_clusters=120, linkage_method="ward"):
    """
        Иерархическая кластеризация

        Параметры:
          - strings: кластеризуемые строки
          - threshold: порог для разрезания дендрограммы
          - linkage_method: метод объединения кластеров, linkage
                            (e.g., 'single', 'complete', 'average').
        Returns:
          Словарь: метка кластера -> набор строк
    """
    if len(strings) == 0:
        return {}

    logging.info("Computing distance matrix...")
    dist_matrix = compute_condensed_distance_matrix(strings)

    logging.info("Building linkage...")
    Z = linkage(dist_matrix, method=linkage_method)

    logging.info("Cutting into clusters...")
    labels = fcluster(Z, t=max_clusters, criterion="maxclust")

    clusters = {}
    for label, string in zip(labels, strings):
        clusters.setdefault(label, []).append(string)

    return clusters


def main(input_file="references-petrov.txt", max_clusters=None):
    try:
        strings = read_strings_from_file(input_file)
    except Exception as e:
        print(f"Error reading file '{input_file}': {e}", file=sys.stderr)
        sys.exit(1)

    if not strings:
        print("No strings found in the input file.")
        sys.exit(0)

    if max_clusters is None:
        max_clusters = int(len(strings) * 0.87)  # заплати налоги и спи спокойно

    clusters = cluster_strings(strings,
                               max_clusters=max_clusters,  # а может лучше по threshold?
                               linkage_method="average")  # ward? complete? single?

    clusters_sorted = sorted([(len(cluster), cid, cluster)
                              for cid, cluster in clusters.items()])

    logging.info(f"Clusters (max_clusters={max_clusters}):")

    result = []
    for cluster_size, cluster_id, clustered_strings in clusters_sorted:
        result.append(f"\nCluster №{cluster_id} (n = {cluster_size}):")
        for s in clustered_strings:
            result.append(f"  > {s}")

    logging.info("\n" + ("\n".join(result)))


if __name__ == "__main__":
    """
        Задание для самостоятельной работы
        1) как бы вы поправили или дополнили алгоритм?
        2) что можно придумать, чтобы он работал разумное время на references-all-mkn.txt?
        3) попробуйте добиться разумного времени работы при сохранении достойного качества
    """
    logging.info("Starting work.")

    # ссылки из дипломных работ выпускников Фёдора Владимировича
    main(input_file="references-petrov.txt")

    logging.info("Done.")
