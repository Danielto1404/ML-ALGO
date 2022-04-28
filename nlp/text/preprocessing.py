import cmath
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix


def document_word_generator(documents):
    for i, d in enumerate(documents):
        for w in d:
            yield i, w


def skip_none(func):
    def check_none(item):
        if item is not None:
            func(item)


class TextFitTransform:
    def fit(self, documents):
        raise NotImplementedError

    def transform(self, documents):
        raise NotImplementedError

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)


class WordsCounter(TextFitTransform):
    def __init__(self):
        self.count_matrix = None
        self.df = None
        self.number_of_documents = None
        self.terms_indices = None

    def __sparse_values__(self, tf):
        return list(tf.values())

    def count(self, documents, fill_indexer=True) -> (csr_matrix, defaultdict):
        tf = defaultdict(int)
        index = 0

        if fill_indexer:
            self.terms_indices = {}
            self.df = defaultdict(set)

        for document, word in document_word_generator(documents):
            if fill_indexer and word not in self:
                self.terms_indices[word] = index
                self.df[word].add(document)
                index += 1

            term_index = self.terms_indices.get(word)
            if term_index is not None:
                tf[(document, term_index)] += 1

        data = self.__sparse_values__(tf)

        return csr_matrix((data, zip(*tf)),
                          shape=(len(documents), self.number_of_words)).tocsr()

    def fit(self, documents):
        self.number_of_documents = len(documents)
        self.count_matrix = self.count(documents)

    def transform(self, documents):
        return self.count(documents, fill_indexer=False)

    @property
    def number_of_words(self):
        return len(self.terms_indices)

    def __contains__(self, item):
        if not isinstance(item, str):
            raise TypeError('Item {} should be of type string'.format(item))

        return item in self.terms_indices


class TfIdf(WordsCounter):
    def __sparse_values__(self, tf):
        tfidf = np.array([cmath.log(self.number_of_documents / len(self.df[term])) for _, term in tf])
        return tfidf

    @property
    def tfidf(self):
        """
        Alias for calling count matrix

        :return: self.count_matrix
        """
        return self.count_matrix


print(TfIdf().fit_transform([['aa'], ['bb'], ['aa']]))
