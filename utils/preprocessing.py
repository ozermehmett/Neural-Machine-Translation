import re
import unicodedata
import tensorflow as tf
from tensorflow import keras
from language import Lang

def unicodeToAscii(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                  if unicodedata.category(c) != "Mn" or c in "çğıöşüÇĞİÖŞÜ")

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([!.?])", r" \1", s)
    s = re.sub(r"[^a-zA-ZçğıöşüÇĞİÖŞÜ?.!]+", " ", s)
    s = re.sub(r"(cc by [^.]+|attribution[^.]+|tatoeba\.org[^.]+)", "", s)
    return s

def load_dataset(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    pairs = [[normalizeString(line.split('\t')[0]),
              normalizeString(line.split('\t')[1])] for line in lines]
    return pairs

def filterPair(p, max_length):
    return len(p[0].split()) < max_length and len(p[1].split()) < max_length

def filterPairs(pairs, max_length):
    return [pair for pair in pairs if filterPair(pair, max_length)]

def sentencetoIndexes(sentence, lang):
    indexes = [lang.word2int[word] for word in sentence.split()]
    indexes.append(1)  # EOS_token
    return indexes

def prepare_data(pairs, lang1, lang2, max_length=10):
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    for pair in pairs:
        input_lang.addSentence(pair[1])
        output_lang.addSentence(pair[0])

    input_seq = []
    output_seq = []

    for pair in pairs:
        input_seq.append(sentencetoIndexes(pair[1], input_lang))
        output_seq.append(sentencetoIndexes(pair[0], output_lang))

    input_tensor = keras.preprocessing.sequence.pad_sequences(
        input_seq, maxlen=max_length, padding='post', truncating='post')
    output_tensor = keras.preprocessing.sequence.pad_sequences(
        output_seq, padding='post', truncating='post')

    return input_tensor, output_tensor, input_lang, output_lang
