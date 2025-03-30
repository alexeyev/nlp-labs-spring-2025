"""
Надо скачать и распаковать вот это:
 https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/Digital_Music.json.gz
"""

import json

from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm

if __name__ == "__main__":

    reviews = [json.loads(line).get("reviewText", None)
               for line in open("Digital_Music.json", "r", encoding="utf-8")]

    with open("tmp-train.txt", "w", encoding="utf-8") as wf:
        for review in tqdm(reviews, "reviews"):
            review = review.strip()

            if review is None or not review:
                continue

            sentences = sent_tokenize(review, language="english")

            for sentence in sentences:
                toks = [w for w in word_tokenize(sentence.lower()) if str.isalpha(w)]
                wf.write(" ".join(toks) + "\n")
