

if __name__ == "__main__":
    """
        В этом примере мы учим рекуррентный теггер на отдельных предложениях
          в НАСТОЯЩЕЙ ЖИЗНИ, конечно, надо добивать предложения до большей длины,
          чтобы обучать на минибатчах.
        
        Попробуйте -- здесь есть масса способов улучшить результаты! И архитектурно,
        и элементарным выбором гиперпараметров -- и так далее.
        
        Ну и не надо забывать, что смотреть на качество и ошибки на тестовой выборке, 
        не используя для этого девсет -- нельзя, иначе получите бессмысленную модель
    """
    import os
    import logging
    from gensim.models.fasttext import load_facebook_vectors
    from gensim.models import Word2Vec

    from model_lstm import LSTMTagger
    from utils import read_data, write_data

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/lstm_wv.log', mode='a', encoding='utf-8')
        ]
    )

    os.makedirs("output", exist_ok=True)

    logging.debug("Reading data")
    train_sents = read_data("data/train.conll")
    test_sents = read_data("data/test.conll")

    X_train = [[w for w, label in sent] for sent in train_sents]
    y_train = [[label for w, label in sent] for sent in train_sents]

    X_test = [[w for w, label in sent] for sent in test_sents]
    y_test = [[label for w, label in sent] for sent in test_sents]

    tag_to_ix = {"B-WoA": 1, "I-WoA": 2, "B-Artist": 3, "I-Artist": 4, "O": 0}

    EMBEDDING_DIM = 256
    HIDDEN_DIM = 128
    LEARNING_RATE = 0.1
    EPOCHS = 30
    LAYERS = 1
    IS_FASTTEXT = True

    if not IS_FASTTEXT:
        emb_model = Word2Vec.load("word2vec100e.bin").wv
    else:
        emb_model = load_facebook_vectors(path="fasttext-custom100e.bin",
                                          encoding="utf-8")

    model = LSTMTagger(embedding_dim=EMBEDDING_DIM,
                       hidden_dim=HIDDEN_DIM,
                       tagset_size=len(tag_to_ix),
                       emb_model=emb_model,
                       label_mapping=tag_to_ix,
                       num_lstm_layers=LAYERS,
                       is_fasttext=IS_FASTTEXT)

    model.fit(X_train, y_train, epochs=EPOCHS, learning_rate=LEARNING_RATE)
    y_pred, pred = model.predict(X_test), []

    for x_seq, y_seq in zip(test_sents, y_pred):
        pred.append(list(zip([w for w, _ in x_seq], y_seq)))

    assert len(pred) == len(y_pred)
    assert len(pred) == len(test_sents)

    logging.debug("Saving to file")
    write_data(pred, f"output/pred-lstm-{'ft' if IS_FASTTEXT else 'wv'}.conll")
