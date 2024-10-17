import json

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2int = {}
        self.word2count = {}
        self.int2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addWord(self, word):
        if word not in self.word2int:
            self.word2int[word] = self.n_words
            self.word2count[word] = 1
            self.int2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def save(self, filename):
        data = {
            'name': self.name,
            'word2int': self.word2int,
            'word2count': self.word2count,
            'int2word': self.int2word,
            'n_words': self.n_words
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        lang = cls(data['name'])
        lang.word2int = data['word2int']
        lang.word2count = data['word2count']
        lang.int2word = {int(k): v for k, v in data['int2word'].items()}
        lang.n_words = data['n_words']
        return lang
