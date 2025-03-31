"""
    Based on:
    https://github.com/scrapinghub/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb
"""
import logging
from typing import List, Iterable, Tuple, Any

from utils import read_data, write_data


def word2features(sent: List[Tuple[str, Any]], i: int):
    """ Задаём признаки для i-й позиции в предложении """
    word = sent[i][0]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        # 'word.isupper=%s' % word.isupper(),
        # 'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
    ]
    if i > 0:
        word1 = sent[i - 1][0]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            # '-1:word.istitle=%s' % word1.istitle(),
            # '-1:word.isupper=%s' % word1.isupper(),
        ])
    else:
        features.append('BOS')

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            # '+1:word.istitle=%s' % word1.istitle(),
            # '+1:word.isupper=%s' % word1.isupper(),
        ])
    else:
        features.append('EOS')

    return features


def sent2features(sent):
    """ Строим признаки по всему предложению """
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    """ Преобразуем пары токен-метка в список меток """
    return [label for token, label in sent]


def sent2tokens(sent):
    """ Преобразуем пары токен-метка в список токенов """
    return [token for token, label in sent]


def print_transitions(trans_features: Iterable[Tuple[Tuple[str, str], float]]):
    for (label_from, label_to), weight in trans_features:
        logging.info("%-9s -> %-9s %0.4f" % (label_from, label_to, weight))


def print_state_features(state_features: Iterable[Tuple[Tuple[str, str], float]]):
    for (attr, label), weight in state_features:
        logging.info("%0.4f %-8s %s" % (weight, label, attr))


if __name__ == "__main__":
    import os
    import pycrfsuite

    from collections import Counter

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Logs to console
            logging.FileHandler('logs/crf.log', mode='a', encoding='utf-8')
        ]
    )

    os.makedirs("output", exist_ok=True)

    logging.debug("Reading data")
    train_sents = read_data("data/train.conll")
    test_sents = read_data("data/test.conll")

    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,  # L1
        'c2': 0.01,  # L2
        'max_iterations': 100,  # Остановка

        # Не наблюдаемые, но возможные переходы
        'feature.possible_transitions': True
    })

    logging.debug("Training...")
    trainer.train("my-music-model.crf", )

    logging.debug(str(trainer.logparser.last_iteration))

    tagger = pycrfsuite.Tagger()
    tagger.open("my-music-model.crf")

    y_pred, pred = [tagger.tag(xseq) for xseq in X_test], []

    for x_seq, y_seq in zip(test_sents, y_pred):
        pred.append(list(zip([w for w, _ in x_seq], y_seq)))

    assert len(pred) == len(y_pred)
    assert len(pred) == len(test_sents)

    logging.debug("Saving to file")
    write_data(pred, "output/pred-crf.conll")
    info = tagger.info()

    logging.info("Top likely transitions:")
    print_transitions(Counter(info.transitions).most_common(5))

    logging.info("Top unlikely transitions:")
    print_transitions(Counter(info.transitions).most_common()[-5:])

    logging.info("Top positive:")
    print_state_features(Counter(info.state_features).most_common(5))

    logging.info("Top negative:")
    print_state_features(Counter(info.state_features).most_common()[-5:])
