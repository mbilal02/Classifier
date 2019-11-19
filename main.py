#from algorithm.model import Model, NER
from keras import Input, Model
from utils.util import ReadFile
max_seq_length = 100  # Your choice here.
global max_seq_length
import tensorflow_hub as hub
import keras.backend as K
from keras.layers import Layer, Bidirectional, Dense, LSTM


class BERT(Layer):
    def __init__(self, output_representation=None, **kwargs):
        self.bert = None
        super(BERT, self).__init__(**kwargs)

        if output_representation:
            self.output_representation = 'sequence_output'
        else:
            self.output_representation = 'pooled_output'

    def build(self, input_shape):
        self.bert = hub.Module('https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1',
                               trainable=True, name="{}_module".format(self.name))

        self.trainable_weights += K.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(BERT, self).build(input_shape)

    def call(self, x, mask=None):
        return self.bert(dict(input_ids=x[0], input_mask=x[1], segment_ids=x[2]), as_dict=True, signature='tokens')[self.output_representation]

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (None, 768)

def ner_classifier():
    file = ReadFile()
    X, x_tr, x_val, x_ts, y_tr, y_val, y_ts = file.wrapper_sequences()
    x_tr = sequence_helper(x_tr, 100)
    x_val = sequence_helper(x_val, 100)
    x_ts = sequence_helper(x_ts, 100)
    word_input = Input(shape=(100,), dtype="string")
    embedding =BERT(output_representation='sequence_output') (word_input)
    rnn = Bidirectional(LSTM(units=256, activation='tanh', return_sequences=True))(embedding)
    prediction_layer = (Dense(units=9, activation='softmax'))(rnn)
    model = Model(inputs=word_input, outputs=prediction_layer)
    import numpy as np
    y_tr = np.array(y_tr)
    y_val = np.array(y_val)
    y_ts = np.array(y_ts)
    print(y_tr.shape)
    y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1],1)
    y_val = y_val.reshape(y_val.shape[0], y_val.shape[1],1)
    y_ts = y_ts.reshape(y_ts.shape[0], y_ts.shape[1],1)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(np.array(x_tr), np.array(y_tr), validation_data=[np.array(x_val), np.array(y_val)], batch_size=50, epochs=20)
    prediction = model.predict(np.array(x_ts), np.array(y_ts), verbose=1)

    prediction = np.argmax(prediction, axis=-1)
    print(prediction[0])

if __name__ == '__main__':
    ner_classifier()


#


