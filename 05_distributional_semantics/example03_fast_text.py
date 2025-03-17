import logging

from gensim.models import FastText

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler('logs/fasttext.log', mode='a', encoding='utf-8')
    ]
)

if __name__ == "__main__":
    import os

    if not os.path.exists("sentences.txt"):
        print("Please run `prepare_data.py` first. "
              "Make sure you have a stable Internet connection.")

    with open("sentences.txt", "r", encoding="utf-8") as rf:
        sentences = [line.strip().split() for line in rf]

    # Обучение запускается прямо в конструкторе
    model = FastText(sentences=sentences,
                     window=5,
                     min_count=4,
                     workers=6,
                     sg=1,
                     negative=25,
                     epochs=15)

    sim = model.wv.most_similar

    for word in ["красный", "яблоко", "пушкин"]:
        logging.debug(f'{" ".join([f"{w}:{s:.3f}" for w, s in sim(positive=[word], topn=5)])}\n')