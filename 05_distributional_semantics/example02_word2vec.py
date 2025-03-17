import os
import logging

from gensim.models import Word2Vec

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler('logs/word2vec.log', mode='a', encoding='utf-8')
    ]
)

if __name__ == "__main__":
    from tqdm import tqdm

    """
        Спорный тейк: для небольших массивов (коллекций, наборов) текстов
            лучше использовать skip-gram (sg=1)
            https://code.google.com/archive/p/word2vec/#Performance
        
        Существенное значение имеют все параметры, заданные ниже,
            однако, самое главное -- размер и разнообразие обучающей выборки
        
        Поэкспериментируйте с разными параметрами и убедитесь "на глазок",
            что результаты на `sentences.txt` при прочих равных куда хуже 
            результатов на `sentences-large.txt` (но всё равно так себе)
        
        Для численной оценки качества можно использовать RUSSE, попробуйте
            (но поправьте датасет: там кое-где есть латинские буквы 'c' и 'o' 
            вместо кириллических, это баг)!
            https://github.com/nlpub/russe-evaluation
    """

    if not os.path.exists("sentences.txt"):
        print("Please run `prepare_data.py` first. "
              "Make sure you have a stable Internet connection.")

    with open("sentences.txt", "r", encoding="utf-8") as rf:
        sentences = [[t for t in line.strip().split()] for line in tqdm(rf)]

    # Обучение запускается прямо в конструкторе
    model = Word2Vec(sentences=sentences,
                     window=5,
                     min_count=4,
                     workers=3,
                     sg=1,
                     negative=25,
                     epochs=170)

    seed_words = ["красный", "яблоко", "пушкин"]

    for query in seed_words:
        print(query)
        for word, s in model.wv.most_similar(positive=[query], topn=5):
            print(f" > {word} ({s:.3f})")
