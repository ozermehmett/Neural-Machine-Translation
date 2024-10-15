from constans import EOS_TOKEN, SOS_TOKEN


class Lang(object):
    def __init__(self, name):
        self.name = name
        self.word2int = {}
        self.word2count = {}
        self.int2word = {SOS_TOKEN : "SOS", EOS_TOKEN : "EOS"}
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
