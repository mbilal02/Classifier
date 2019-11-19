from nltk.tokenize.treebank import TreebankWordDetokenizer

from algorithm.layer import DeepLearner
import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras.models import Model
#from bert.tokenization.albert_tokenization import FullTokenizer
from algorithm.tokenizer import FullTokenizer
from utils.util import ReadFile
import keras as keras
import numpy as np
import logging as logger
log= logger.getLoggerClass()

class NER:

    def get_features(self,list_name, max_seq_length):


        #max_seq_length = 128  # Your choice here.
        #global max_seq_length
        #log.info("Input is running")
        input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=np.int32,
                                               name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=np.int32,
                                           name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=np.int32,
                                            name="segment_ids")
        #log.info("Initializing bert model")
        bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                                    trainable=True)
        #log.info("creating bert_layer")
        pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

        bert_model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])
        bert_model.summary()
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        #tokenizer = FullTokenizer(vocab_file, do_lower_case)
        ids, masks, segments = self.get_data_ids( vocab_file,do_lower_case, list_name)
        pool_embs, all_embs = bert_model.predict([ids, masks, segments])

        return pool_embs, all_embs


    def bert_masks(self,list_name, tokenizer, max_seq_length):
        if len(list_name) > max_seq_length:
            raise IndexError("Token length more than max seq length!")
        return [1] * len(list_name) + [0] * (max_seq_length - len(list_name))

    def get_segments(self,list_name, tokenizer, max_seq_length):
        """Segments: 0 for the first sequence, 1 for the second"""
        if len(list_name) > max_seq_length:
            raise IndexError("Token length more than max seq length!")
        segments = []
        current_segment_id = 0
        for token in list_name:
            segments.append(current_segment_id)
            if token == "[SEP]":
                current_segment_id = 1
        return segments + [0] * (max_seq_length - len(list_name))


    def bert_ids(self, list_name, tokenizer, max_seq_length):
        token_ids = tokenizer.convert_tokens_to_ids(list_name)
        input_ids = token_ids + [0] * (max_seq_length - len(token_ids))
        return input_ids

    def get_data_ids(self,vocab, flag, list_name):
        max_len = 100
        sentences = []
        tokenizer = FullTokenizer(vocab, flag)
        for sent in list_name:
           detoc =  [TreebankWordDetokenizer().detokenize(sent)]
           stokens = tokenizer.tokenize(detoc[0])
           sentences.append(stokens)
        ids = list()
        masks = list()
        segments= list()

        for sentence in sentences:
            stokens = ["[CLS]"] + sentence + ["[SEP]"]
            ids.append(self.bert_ids(stokens, tokenizer, max_len))
            masks.append(self.bert_masks(stokens, tokenizer, max_len))
            segments.append(self.get_segments(stokens, tokenizer, max_len))
        return ids, masks, segments