from typing import List, Tuple


def read_data(path: str = "data/train.conll") -> List[List[Tuple[str, str]]]:
    """ Зачитываем данные в формате conll2003 """

    sentences = [[]]

    with open(path, "r", encoding="utf-8") as rf:
        for line in rf:
            if not line.strip():
                sentences.append([])
            else:
                token, tag = line.strip().split("\t")
                sentences[-1].append((token, tag))
    return sentences[:-2]


def write_data(data: List[List[Tuple[str, str]]],
               path: str = "output/pred.conll") -> None:
    """ Сохраняем результаты разметки в формате conll2003 """

    with open(path, "w", encoding="utf-8") as wf:
        for pairs in data:
            for w, t in pairs:
                wf.write(f"{w}\t{t}\n")
            wf.write("\n")
        wf.write("\n")


if __name__ == "__main__":
    train_set = read_data()
    test_set = read_data("data/test.conll")

    assert len(train_set) == 1950, f"Что-то не то с размером обучающей выборки: {len(train_set)}"
    assert len(test_set) == 599, f"Что-то не то с размером тестовой выборки: {len(test_set)}"
