from nlp.core.char_lstm import _CharLSTM


class CharLSTM:
    def __init__(self, sentences, hidden_size):
        self.model = _CharLSTM(vocab_size=0, hidden_size=hidden_size)

    def train(self, epochs):
        pass
