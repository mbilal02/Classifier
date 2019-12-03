import csv
from collections import Counter

import keras as keras
from nltk.corpus.reader.conll import ConllCorpusReader

sent_max = 100


class ReadFile:

    def fileReader(self, filename):

        '''
        Data reader for CoNLL format data
        '''
        root = "data/"
        sentences = []

        ccorpus = ConllCorpusReader(root, ".conll", ('words', 'pos', 'tree'))

        tagged = ccorpus.tagged_sents(filename)

        return tagged



    def process(self, sentences):
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

        return {
        'I-tvshow': 20,
        'B-tvshow': 19,
        'I-sportsteam': 18,
        'B-sportsteam': 17,
        'I-product': 16,
        'B-product': 15,
        'I-person': 14,
        'B-person': 13,
        'I-other': 12,
        'B-other': 11,
        'I-musicartist': 10,
        'B-musicartist': 9,
        'I-movie': 8,
        'B-movie': 7,
        'I-geo-loc': 6,
        'B-geo-loc': 5,
        'I-facility': 4,
        'B-facility': 3,
        'I-company': 2,
        'B-company': 1,
        'O': 0}


    def get_masks(self,tokens, max_seq_length):
        """Mask for padding"""
        if len(tokens) > max_seq_length:
            raise IndexError("Token length more than max seq length!")
        return [1] * len(tokens) + [0] * (max_seq_length - len(tokens))

    # raise NotImplementedError

    def get_segments(self, tokens, max_seq_length):
        """Segments: 0 for the first sequence, 1 for the second"""
        if len(tokens) > max_seq_length:
            raise IndexError("Token length more than max seq length!")
        segments = []
        current_segment_id = 0
        for token in tokens:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (max_seq_length - len(tokens))

    def get_ids(self, tokens, tokenizer, max_seq_length):
        """Token ids from Tokenizer vocab"""
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
        return input_ids

    # def create_matrices(self, sentences):
    #   x, y = self.process(sentences)
    # TODO: implement above pad_xSequences and then call that function here to pad x equal to the sentence max_len
    #   global sent_max
    # y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=sent_max, padding='post')

    # import numpy as np

    # x = np.array(x)

    # y = np.array(y)

    #  return x, y
    def wrapper_sequences(self):
        all_sents = list()
        files = ['train_2016', 'dev_2016', 'test_2016']
        #root = 'data/'
        train = self.fileReader(files[0])
        dev = self.fileReader(files[1])
        test = self.fileReader(files[2])
        x_tr, y_tr = self.process(train)
        x_val, y_val = self.process(dev)
        x_ts, y_ts = self.process(test)
        import tensorflow as tf
        y_tr =tf.keras.preprocessing.sequence.pad_sequences(y_tr, maxlen=100, padding='post')
        y_val = tf.keras.preprocessing.sequence.pad_sequences(y_val, maxlen=100, padding='post')
        y_ts = tf.keras.preprocessing.sequence.pad_sequences(y_ts, maxlen=100, padding='post')

        X = x_tr + x_val + x_ts
        return X, x_tr, x_val, x_ts, y_tr, y_val, y_ts

    def encode(self, X):
        words = list()
        for sentence in X:
            for word in sentence:
                words.append(word)
        word_counts = Counter(words)
        vocab_inv = [x[0] for x in word_counts.most_common()]
        vocab = {x: i + 1 for i, x in enumerate(vocab_inv)}
        id_to_vocb = {i: x for x, i in vocab.items()}
        return vocab

    def encoding_sequences(self,X,sentences):
        vocab = self.encode(X)
        sents = list()
        for sentence in sentences:
            sent= list()
            for word in sentence:
                sent.append(vocab[word])
            sents.append(sent)
        return sents

    def model_data(self):
        X, x_tr, x_val, x_ts, y_tr, y_val, y_ts = self.wrapper_sequences()
        X_train= self.encoding_sequences(X, x_tr)
        X_val = self.encoding_sequences(X, x_val)
        X_test = self.encoding_sequences(X, x_ts)

        return X_train, X_val, X_test, y_tr, y_val, y_ts

    def sequence_helper(self,x_in, sent_maxlen):
        '''
        Helper function for word sequences (text data sepcific)
        :param x_in:
        :param sent_maxlen:
        :return: Word sequences
        '''

        new_X = []
        for seq in x_in:
            new_seq = []
            for i in range(sent_maxlen):
                try:
                    new_seq.append(seq[i])
                except:
                    new_seq.append('__pad__')
            new_X.append(new_seq)
        return new_X


    def getLabels(self,y_test, vocabulary):
        '''
        Maps integer to the label map
        '''
        #
        classes = []
        # y = np.array(y_test).tolist()
        for i in y_test:
            label = []
            pre = [[k for k, v in vocabulary.items() if v == j] for j in i]
            for i in pre:
                for j in i:
                    label.append(j)
            classes.append(label)
        return classes

    def write_f(self,filename, dataset, delimiter='\t'):
        """dataset is a list of tweets where each token can be a tuple of n elements"""
        with open(filename, '+w', encoding='utf8') as stream:
            writer = csv.writer(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE, quotechar='')

            for tweet in dataset:
                writer.writerow(list(tweet))
    def save_predictions(self,filename, tweets, labels, predictions):
        """save a file with token, label and prediction in each row"""
        dataset, i = [], 0
        for n, tweet in enumerate(tweets):
            tweet_data = list(zip(tweet, labels[n], predictions[n]))
            dataset += tweet_data + [()]
        self.write_f(filename, dataset)

