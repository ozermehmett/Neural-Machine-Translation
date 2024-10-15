from constants import MAX_LENGTH
from lang import Lang
from utils import sentencetoIndexes

import keras



def build_lang(lang1, lang2, max_length=MAX_LENGTH, pairs):
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    input_seq = []
    output_seq = []

    for pair in pairs:
        input_lang.addSentence(pair[1])
        output_lang.addSentence(pair[0])
    for pair in pairs:
        input_seq.append(sentencetoIndexes(pair[1], input_lang))
        output_seq.append(sentencetoIndexes(pair[0], output_lang))
    return keras.preprocessing.sequence.pad_sequences(input_seq, 
                                                      maxlen=max_length, 
                                                      padding='post',
                                                      truncating='post'), 
                                                      keras.preprocessing.sequence.pad_sequences(output_seq, 
                                                                                                 padding='post', 
                                                                                                 truncating='post'),
                                                      input_lang, 
                                                      output_lang
