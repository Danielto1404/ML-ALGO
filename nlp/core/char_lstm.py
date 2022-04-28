import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange

from nlp.core.vocabs import CharVocabulary

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class _CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, x) -> torch.tensor:
        """
        :param x: (batch_size, sequence_size, input_size)
        :return:  (batch_size, hi)
        """
        outputs, _ = self.lstm(x)
        return outputs


class __CharLSTMTrainer:
    def __init__(self, epochs, hidden_size, vocab: CharVocabulary):
        self.model: _CharLSTM = _CharLSTM(vocab_size=vocab.size, hidden_size=hidden_size)
        self.epochs: int = epochs
        self.vocab: CharVocabulary = vocab
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-3)

    def train(self, sentences, batch_size=1):
        dataset = DataLoader(dataset=sentences, shuffle=True, batch_size=batch_size)
        progress = trange(self.epochs, desc='Epoch')
        for _ in progress:
            for sentences_batch in dataset:
                train, target, lens = self.vocab.to_train_target(sentences_batch)
                output = self.model(train)
                self.optimizer.zero_grad()
                error = self.loss(torch.swapaxes(output, 1, 2), target)
                error.backward()
                self.optimizer.step()
                loss = error.detach().item()
                progress.set_postfix_str('Loss: {}'.format(loss))


#
with open('../../data/PRN+PreposAdj+V/rus_text.txt', 'r') as f:
    sentences = f.readlines()

vocab = CharVocabulary(sentences=sentences)
start = vocab.train_one_hot([vocab.get_indices('Мы')], append_start=False)
m = torch.load('lstm_trained.pt')
with torch.no_grad():
    output = m(start).squeeze(axis=0)
print(output.shape)
print(output[-1].argmax())
print(output[-1])
print(vocab.get_char(output[-1].argmax().item()))

# trainer = __CharLSTMTrainer(10, vocab.size, vocab)
# trainer.train(sentences, batch_size=32)
# torch.save(trainer.model, 'lstm_trained.pt')
