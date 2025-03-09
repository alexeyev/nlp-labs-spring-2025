import logging
from functools import lru_cache
from typing import List

import requests
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from pymorphy2 import MorphAnalyzer
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler('logs/downloader.log', mode='a', encoding='utf-8')
    ]
)

Lemmatizer = MorphAnalyzer()


@lru_cache(1000000)
def cached_parse(word):
    """
        Not the best option for Russian lemmatization,
          but it's fast, and context-independence allows to cache stuff
    """
    return Lemmatizer.parse(word)[0].normal_form


def download_and_process_text(url="http://www.lib.ru/PROZA/DOMBROWSKIJ/faculty.txt_Ascii.txt"):
    tokenizer = Tokenizer13a()

    logging.info("Downloading text")
    response = requests.get(url)
    response.encoding = response.apparent_encoding

    if response.status_code != 200:
        raise Exception(f"Failed to download text. Status code: {response.status_code}")

    if "Домбровский" not in response.text:
        logging.warning("бНОПНЯ detected.")
        text = response.content.decode("koi8-r")
    else:
        try:
            text = response.content.decode("utf-8")
        except UnicodeDecodeError:
            logging.warning("This should not happen, trying windows-1251")
            text = response.content.decode("windows-1251")

    logging.debug("First chars:", text[:100])

    text = text.replace("\n\n", "\n")
    sentences: List[str] = sent_tokenize(text, language="russian")
    normalized_sentences: List[List[str]] = []

    for sentence in tqdm(sentences, "sentences"):
        normalized_sentences.append([])
        for token in tokenizer(sentence).split(" "):
            if token.isalnum():
                normalized_sentences[-1].append(cached_parse(token))

    return normalized_sentences


if __name__ == "__main__":

    sentences = download_and_process_text()
    logging.info(f"Processed {len(sentences)} sentences:")

    for i, sent in enumerate(sentences[:15], 1):
        logging.debug(f"{i}. {sent}")

    with open("sentences.txt", "w", encoding="utf-8") as wf:
        wf.write("\n".join([" ".join(s) for s in sentences]))

    sentences_keeper = download_and_process_text("https://lib.ru/PROZA/DOMBROWSKIJ/keeper.txt_Ascii.txt")

    for i, sent in enumerate(sentences_keeper[:5], 1):
        logging.debug(f"{i}. {sent}")

    sentences_monkey = download_and_process_text("https://lib.ru/PROZA/DOMBROWSKIJ/dombrovsky2.txt_Ascii.txt")

    for i, sent in enumerate(sentences_monkey[:5], 1):
        logging.debug(f"{i}. {sent}")

    with open("sentences-larger.txt", "w", encoding="utf-8") as wf:
        wf.write("\n".join([" ".join(s) for s in sentences]))
        wf.write("\n".join([" ".join(s) for s in sentences_keeper]))
        wf.write("\n".join([" ".join(s) for s in sentences_monkey]))

    logging.info("Done.")
