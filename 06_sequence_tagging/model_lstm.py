import logging
from typing import List, Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import KeyedVectors
from tqdm import tqdm

torch.manual_seed(1)


class LSTMTagger(nn.Module):

    def __init__(self,
                 embedding_dim: int,
                 hidden_dim: int,
                 tagset_size: int,
                 emb_model: KeyedVectors,
                 label_mapping: Dict[str, int],
                 default_lr: float = 0.1,
                 num_lstm_layers: int = 2,
                 is_fasttext: bool = False):

        super(LSTMTagger, self).__init__()

        self.hidden_dim: int = hidden_dim
        self.pretr_emb_dim: int = emb_model.vector_size
        self.word_embeddings = nn.Linear(self.pretr_emb_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            bidirectional=True,
                            num_layers=num_lstm_layers)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.fixer = nn.RNN(tagset_size, tagset_size)

        self.emb_model: KeyedVectors = emb_model

        # задаём, как преобразовывать метки в их индексы и наоборот
        self.label2idx: Dict[str, int] = label_mapping
        self.idx2label: Dict[int, str] = {i: l for l, i in self.label2idx.items()}

        # для обучения; строго говоря, эти штуки
        #  обычно лучше задавать не в наследнике nn.Module
        self.loss_function: nn.NLLLoss = nn.NLLLoss()
        self.optimizer: optim.Optimizer = optim.SGD(self.parameters(),
                                                    lr=default_lr)
        self.is_fasttext: bool = is_fasttext

    def word_emb(self, x: str) -> np.ndarray:
        if self.is_fasttext or x in self.emb_model:
            return self.emb_model.get_vector(x)
        else:
            return np.zeros(self.emb_model.vector_size)

    def forward(self, sentence: List[str]):

        with torch.no_grad():
            seq_len: int = len(sentence)
            embedded_sentence = np.array([self.word_emb(w) for w in sentence])

        ext_embeddings = torch.tensor(embedded_sentence, dtype=torch.float)

        # применяем линейный слой к каждому эмбеддингу
        # [S x E]
        embeds = self.word_embeddings(ext_embeddings)

        # получаем выходы LSTM
        # [S x H]
        lstm_out, _ = self.lstm(embeds.view(seq_len,  # длина последовательности
                                            1,  # длина батча
                                            -1
                                            ))

        # получаем скоры по числу меток (применяем к каждой строке линейный слой)
        # [S x 5]
        tag_space: torch.FloatTensor = self.hidden2tag(lstm_out.view(seq_len, -1))

        # we now want to add a 'correcting RNN'
        #   that would 'zero out' illegal label combinations
        prelim_predictions = F.softmax(tag_space, dim=1).view(seq_len, 1, -1)
        fixer_out, _ = self.fixer(prelim_predictions)
        fixer_gates = F.sigmoid(fixer_out).view(seq_len, -1)

        fixed_scores = fixer_gates * tag_space

        # преобразовываем в предсказания:
        #   для negative log likelihood нужно
        #   подавать логарифмы оценки вероятностей
        # [S x 5]
        output = F.log_softmax(fixed_scores, dim=1)

        return output

    def fit(self, X, y, epochs: int, learning_rate: float):

        self.train()
        self.optimizer.lr = learning_rate

        for _ in tqdm(range(epochs), "epochs"):
            losses = []
            for sentence, tags in zip(X, y):
                # пайторч аккумулирует градиенты, их надо сбрасывать
                self.zero_grad()

                # входы и выходы для обучения
                targets = prepare_sequence(tags, self.label2idx)

                # проход по сетке
                tag_scores = self.__call__(sentence)

                # вычисление невязки и обратное распространение
                loss = self.loss_function(input=tag_scores, target=targets)

                with torch.no_grad():
                    losses.append(loss.detach().numpy())

                loss.backward()
                self.optimizer.step()

            logging.info(f"Loss: {np.mean(losses)}")

        self.eval()

    def predict(self, dataset: Iterable[List[str]]):

        results = []

        with torch.no_grad():
            for sentence in dataset:
                tag_scores: torch.Tensor = self.__call__(sentence)
                tag_ids = torch.argmax(tag_scores, dim=1).detach().numpy()
                assert len(sentence) == tag_ids.shape[0]
                results.append([self.idx2label[tag_idx] for tag_idx in tag_ids])

        return results


def prepare_sequence(seq: List[str], to_ix: Dict[str, int]):
    return torch.tensor([to_ix[w] for w in seq], dtype=torch.long)


if __name__ == "__main__":
    pass
