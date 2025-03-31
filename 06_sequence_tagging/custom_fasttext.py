import os

import fasttext
from gensim.models.fasttext import FastTextKeyedVectors, load_facebook_vectors

unsupervised_default = {
    'model': "skipgram",
    'dim': 50,
    'ws': 5,
    'epoch': 100,
    'minCount': 10,
    'minn': 3,
    'maxn': 6,
    'neg': 5,
    'wordNgrams': 1,
    'loss': "ns",
    'thread': 7,
    'lrUpdateRate': 100000,
    'verbose': 2,
}

# if any([f.startswith("fasttext-custom") for f in os.listdir()]):
#     ft: FastTextKeyedVectors = load_facebook_vectors("fasttext-custom100e.bin")
#     print("Please:", ft.most_similar(positive=["please"]))
#     print("Other:", ft.most_similar(positive=["other"]))
#     print("Aphex:", ft.most_similar(positive=["aphex"]))
#     print("Album:", ft.most_similar(positive=["album"]))
#     print("Song:", ft.most_similar(positive=["song"]))
#     print("Band:", ft.most_similar(positive=["band"]))
#     quit()

model = fasttext.train_unsupervised("tmp-train.txt", **unsupervised_default)
model.save_model("fasttext-custom100e.bin")
