import csv
from collections import Counter

import keras as keras

sent_max = 100


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

        return {'o'+'\n': 0,
                'o':0
            , 'b-per\n': 1
            , 'i-per\n': 2
            , 'b-loc\n': 3
            , 'i-loc\n': 4
            , 'b-misc\n': 5
            , 'i-misc\n': 6
            , 'b-org\n': 7
            , 'i-org\n': 8}

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
        files = ['train.txt', 'valid.txt', 'test.txt']
        root = 'data/'
        for filename in files:
            file = self.fileReader(root + filename)
            file += file
            all_sents.append(file)
        x_tr, y_tr = self.process(all_sents[0])
        x_val, y_val = self.process(all_sents[1])
        x_ts, y_ts = self.process(all_sents[2])
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

