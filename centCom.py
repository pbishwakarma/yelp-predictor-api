from __future__ import print_function
import csv
# import pandas as pd
# import numpy as np
# import os

import os
import numpy as np


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
# from keras.layers import Dense, Input, Flatten
# from keras.layers import Conv1D, MaxPooling1D, Embedding
# from keras.models import Model
from keras.models import load_model
# import sys


class CentCom:

    def __init__(self,  model_file_path, data_path='../data/', file_name='first_500k.csv', label_col='stars',
                 col_names=['stars', 'text', 'cool', 'useful', 'funny']):

        self.data_path = data_path
        self.file_name = file_name

        self.embedding_index = {}

        self.MAX_SEQUENCE_LENGTH = 1000
        self.MAX_NB_WORDS = 20000
        self.EMBEDDING_DIM = 100

        self.texts = []  # list of text samples
        self.labels_index = {}  # dictionary mapping label name to numeric id
        self.labels = []  # list of label ids
        self.text_data = None

        if model_file_path is not None:
            self.model = load_model(model_file_path)  # loads a pre-trained keras model
        else:
            self.model = None

        self.tokenizer = Tokenizer(nb_words=self.MAX_NB_WORDS)
        self.word_index = None

        self.label_col = label_col

    def load_embedding(self, glove_dir='glove.6B/'):
        """ This script loads pre-trained word embeddings (GloVe embeddings).

        Sets the embedding_index attribute

        GloVe embedding data can be found at:
        http://nlp.stanford.edu/data/glove.6B.zip
        (source page: http://nlp.stanford.edu/projects/glove/)
        :type glove_dir: basestring"""

        f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            self.embeddings_index[word] = np.asarray(values[1:], dtype='float32')
        f.close()

    def _build_labels_dict(self, label_names):
        """ builds the labels_index dictionary by getting the number of classes
            from the data. """

        for i in range(len(label_names)):
            self.labels_index[label_names[i]] = i

    def preprocess(self):
        """ Populates the list of text samples and the label ids. Operates on the given data """

        self._build_labels_dict(['one', 'two', 'three', 'four', 'five'])

        with open(self.data_path + self.file_name, 'rb') as csvfile:

            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                self.texts.append(row[1])
                self.labels.append(self.labels_index[row[0]])

        print('Found %s texts.' % len(self.texts))

    def fit_tokenizer(self, texts):
        """ tokenize the given texts, can be either a single text or a list of texts. """
        if not isinstance(texts, str) or not isinstance(texts, list):
            print("The sample you have provided is not correctly formatted. " \
                  "Must be a string or an array of strings.")
            return
        else:
            if isinstance(texts, str):
                texts = [texts]

        self.tokenizer.fit_on_texts(texts)

    def get_sequences(self, texts):

        return self.tokenizer.texts_to_sequences(texts)

    def all_tokenize(self):
        self.tokenizer.fit_on_texts(self.texts)

        sequences = self.tokenizer.texts_to_sequences(self.texts)

        word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        return sequences

    def prep_data(self):
        """ pad the sequences to the maximum length, convert out labels array"""

        self.fit_tokenizer(texts=self.texts)
        sequences = self.get_sequences(self.texts)
        self.text_data = pad_sequences(sequences, maxlen=self.MAX_SEQUENCE_LENGTH)

        self.labels = to_categorical(np.asarray(self.labels))
        print('Shape of data tensor:', self.text_data.shape)
        print('Shape of label tensor:', self.labels.shape)

        # split the data into a training set and a validation set
        indices = np.arange(self.text_data.shape[0])
        np.random.shuffle(indices)
        self.text_data = self.text_data[indices]
        self.labels = self.labels[indices]
        nb_validation_samples = int(self.VALIDATION_SPLIT * self.text_data.shape[0])

        x_train = self.text_data[:-nb_validation_samples]
        y_train = self.labels[:-nb_validation_samples]
        x_val = self.text_data[-nb_validation_samples:]
        y_val = self.labels[-nb_validation_samples:]

        return x_train,y_train, x_val, y_val

    def predict(self, sample):
        """ sample is a string or an array of strings """

        if isinstance(sample, str):
            sample = np.array([sample])
        else:
            sample = np.array(sample)
        sample_sequence = self.get_sequences(sample)
        padded = pad_sequences(sample_sequence, maxlen = self.MAX_SEQUENCE_LENGTH)
        # check shape of array before putting it in the prediction function
        if len(sample.shape) != 2:
            padded = np.reshape(padded, (-1, self.MAX_SEQUENCE_LENGTH))
        elif sample.shape[1] != self.MAX_SEQUENCE_LENGTH:
            padded = np.reshape(padded, (-1, self.MAX_SEQUENCE_LENGTH))

        pred_array = self.model.predict(padded)

        max_indices = np.argmax(pred_array, axis=1)

        for i in xrange(len(max_indices)):
            print("The predicted number of stars for sample %s is %s stars" % (i, max_indices[i] + 1))

        return pred_array

    def train(self, x_train, y_train, x_val, y_val):
        """ Fits the model given training and validation data."""

        if self.model is not None:
            self.model.fit(x_train, y_train, validation_data=(x_val, y_val), nb_epoch=2, batch_size=128)

        else:
            print("You need to instantiate a model or load one from a file before training!")

    def set_model(self, file_path):
        if os.path.isfile(file_path):
            self.model = load_model(filepath=file_path)
        else:
            print("This is not a valid file path %s " % file_path)


# def main():
#
#     classifier = CentCom('../initial_model.h5')
#
#     # 1. preprocess and build text and label lists
#     classifier.preprocess()
#
#     # 2. tokenize all the texts, then given sample
#     classifier.all_tokenize()
#     sample = "I absolutely love this restaurant Food was delicious, service was great I could even bring my dog "
#     classifier.predict(sample)
#
#
#
#
#
# main()

# if __name__ == __main__():
#     main()

    




