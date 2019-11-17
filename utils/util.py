from collections import Counter
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import keras as keras

sent_max = 50


class ReadFile:
    def fileReader(self, filename):
        words = []
        labes = []

        with open(filename, mode='rt', encoding='utf8') as f:
            sentences = []
            sentence = []
            for line in f:
                if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                    if len(sentence) > 0:
                        sentences.append(sentence)
                        sentence = []
                    continue
                splits = line.split(' ')
                sentence.append([splits[0].lower(), splits[-1].lower()])
                words.append([splits[0].lower()])
                labes.append([splits[-1].lower()])
        if len(sentence) > 0:
            sentences.append(sentence)
            # print(sentences)
        #
        # words_counts = Counter(words)
        # vocb_list = [x[0] for x in words_counts.most_common()]
        # vocb = {x: i + 1 for i,x in enumerate(vocb_list)}
        # #vocb[vocb_list]=
        return sentences

    def process(self, sentences):
        print(sentences)
        x = list()
        y = list()
        label_vocab = self.get_label_vocab()
        if sentences != []:
            for sentence in sentences:
                word = list()
                label = list()
                for word_index in sentence:
                    word.append(word_index[0])
                    label.append(label_vocab[word_index[1]])
                x.append(word)
                y.append(label)
        else:
            raise ValueError()
        return x, y

    def get_label_vocab(self):

        return {'O': 0
            , 'B-PER': 1
            , 'I-PER': 2
            , 'B-LOC': 3
            , 'I-LOC': 4
            , 'B-MISC': 5
            , 'I-MISC': 6
            , 'B-ORG': 7
            , 'I-ORG': 8}

    def pad_xSequences(self, sentences):
        """
        padding the of the sentences with the max sentence length
        max lenght of sentence let 100 words
        sentence pad with pad
        :return: list of ist of sentences
        """
        max_length = 120
        if not sentences:
            padded = pad_sequences(sentences, padding='post', maxlen=max_length)

        return padded

    # raise NotImplementedError


def create_matrices(self, sentences):
    x, y = self.process(sentences)
    # TODO: implement above pad_xSequences and then call that function here to pad x equal to the sentence max_len
    global sent_max
    y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=sent_max, padding='post')

    import numpy as np

    x = np.array(x)

    y = np.array(y)

    return x, y
