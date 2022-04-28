from functools import reduce

import numpy as np
import torch


class CharVocabulary:
    def __init__(self, sentences, START='<#START#>', END='<#END#>', PAD='<#PAD#>'):
        self.sentences = sentences
        self.START = START
        self.END = END
        self.PAD = PAD
        self.indices_to_chars = {}
        self.chars_to_indices = {}

        self.build()

    def build(self):
        initial_set = {self.START, self.END, self.PAD}
        chars = reduce(lambda acc, s: acc | set(s), self.sentences, initial_set)
        indices = range(len(chars))
        self.indices_to_chars = dict(zip(indices, chars))
        self.chars_to_indices = dict(zip(chars, indices))
        return self

    def get_char(self, index):
        if index >= self.size:
            raise LookupError('Index should be less than vocab size')
        return self.indices_to_chars[index]

    def get_index(self, char):
        i = self.chars_to_indices.get(char, None)
        if i is None:
            raise LookupError('Char {} not presented in vocabulary. Make sure you run build function.'.format(char))

        return i

    def get_indices(self, sentence):
        return list(map(self.get_index, sentence))

    def one_hot_numpy(self, indices):
        n = len(indices)
        one_hot = np.zeros((n, self.size), dtype=np.int)
        one_hot[np.arange(n), indices] = 1
        return one_hot

    def one_hot_sentence(self, sentence=None, indices=None):
        if indices is None:
            indices = self.get_indices(sentence)
        return self.one_hot_numpy(indices)

    def add_padding(self, indices, max_len):
        PAD_IDX = self.get_index(self.PAD)
        return indices + [PAD_IDX] * (max_len - len(indices))

    def train_one_hot(self, indices, append_start=True):
        if append_start:
            START_IDX = self.get_index(self.START)
            train = np.array([self.one_hot_sentence(indices=[START_IDX] + ind) for ind in indices])
        else:
            train = np.array([self.one_hot_sentence(indices=ind) for ind in indices])
        return torch.from_numpy(train).float()

    def target_vec(self, indices, append_end=True):
        if append_end:
            END_IDX = self.get_index(self.END)
            target = np.array([ind + [END_IDX] for ind in indices])
        else:
            target = np.array(indices)
        return torch.from_numpy(target)

    def to_train_target(self, sentences) -> (torch.tensor, torch.tensor, torch.tensor):
        indices = list(map(self.get_indices, sentences))
        lens = np.array(list(map(len, indices)))

        max_len = np.max(lens)
        indices = list(map(lambda ids: self.add_padding(ids, max_len=max_len), indices))

        return self.train_one_hot(indices), self.target_vec(indices), torch.from_numpy(lens + 1)

    @property
    def size(self):
        return len(self.indices_to_chars)
