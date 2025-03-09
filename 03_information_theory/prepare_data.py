# other options here:
# import spacy
# from spacy import Language
# from pymystem3 import Mystem

import logging
from functools import lru_cache
from typing import List

import requests
from nltk.tokenize import sent_tokenize
from pymorphy2 import MorphAnalyzer
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a
from tqdm import tqdm

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to console
        logging.FileHandler('downloader.log', mode='a', encoding='utf-8')
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

    try:
        text = response.content.decode("koi8-r")
    except UnicodeDecodeError:
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

    logging.info("Done.")
