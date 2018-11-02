from layers import HashEmbedding, ReduceSum
from keras.layers import Input, Dense, Activation, Embedding
from keras.models import Model
import hashlib
import nltk
import keras
import numpy as np
import csv
from keras.callbacks import EarlyStopping
from sklearn.datasets import fetch_20newsgroups
from os import listdir
from os.path import isfile, join
from random import randrange
import random

def get_model(embedding, num_classes):
    input_words = Input([None], dtype='int32', name='input_words')

    x = embedding(input_words)
    x = ReduceSum()([x, input_words])

    x = Dense(50, activation='relu')(x)

    x = Dense(num_classes)(x)
    x = Activation('softmax')(x)
    model = Model(input=input_words, output=x)
    return model

def word_encoder(w, max_idx):
    # v = hash(w) #
    v = int(hashlib.sha1(w.encode('utf-8')).hexdigest(), 16)
    return (v % (max_idx-1)) + 1


if __name__ == '__main__':

    text_file = '..\train.csv'

    positive_file_location = '..\aclImdb\\train\pos'
    negative_file_location = '..\aclImdb\\train\\neg'
    positive_files = [f for f in listdir(positive_file_location) if isfile(join(positive_file_location, f))]
    num_pos_files = len(positive_files)
    file_counter = 0
    negative_files = [f for f in listdir(negative_file_location) if isfile(join(negative_file_location, f))]
    files = positive_files + negative_files

    text_data = []
    textData = []
    label_data = np.zeros(len(files))
    for text in files:
        if file_counter < num_pos_files:
            with open('{}/{}'.format(positive_file_location, text), 'rb') as data:
                text_data.append([str(data.read()), 1])
        else:
            with open('{}/{}'.format(negative_file_location, text), 'rb') as data:
                text_data.append([str(data.read()), 0])
        file_counter = file_counter + 1

    text = []
    for values in text_data:
        text.append(values[0])

    indexes = random.sample(range(0, len(text_data)), len(text_data))

    i = 0
    for row in indexes:
        textData.append(text_data[row][0])
        label_data[i] = text_data[row][1]
        i = i + 1

    use_hash_embeddings = True
    embedding_size = 20
    num_buckets = 5000  # number of buckets in second hashing layer (hash embedding)
    max_words = 10 ** 6  # number of buckets in first hashing layer
    max_epochs = 50
    num_hash_functions = 4

    if use_hash_embeddings:
        embedding = HashEmbedding(max_words, num_buckets, embedding_size, num_hash_functions=num_hash_functions)
    else:
        embedding = Embedding(max_words, embedding_size)

    # create dataset
    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
    num_classes = 2
    #twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    data = [nltk.word_tokenize(text)[0:200] for text in textData]
    # data = [nltk.word_tokenize(text) for text in twenty_train.data]
    data_encoded = [[word_encoder(w, max_words) for w in text] for text in data]
    max_len = max([len(d) for d in data])


    # pad data
    data_encoded = [d+[0]*(max_len-len(d)) for d in data_encoded]

    idx_test = int(len(data_encoded)*0.5)
    data_encoded = np.vstack(data_encoded)
    #targets = np.asarray(twenty_train.target, 'int32').reshape((-1,1))

    model = get_model(embedding, num_classes)
    metrics = ['accuracy']
    loss = 'sparse_categorical_crossentropy'
    model.compile(optimizer=keras.optimizers.Adam(),loss=loss, metrics=['accuracy'])

    print('Num parameters in model: %i' % model.count_params())
    model.fit(data_encoded[0:idx_test,:], label_data[0:idx_test], validation_split=0.1,nb_epoch=max_epochs,
              callbacks=[EarlyStopping(patience=5)])

    test_result = model.test_on_batch(data_encoded[idx_test::, :], label_data[idx_test::])
    for i, (name, res) in enumerate(zip(model.metrics_names, test_result)):
        print('%s: %1.4f' % (name, res))

    results = model.predict(data_encoded[idx_test::, :])

    start = idx_test
    correctResults = 0
    i = 0
    for result in results:
        isDataPositive = True
        if result[0] > 0.5:
            isDataPositive = False
        else:
            isDataPositive = True
        if isDataPositive and label_data[start + i] == 1:
            correctResults = correctResults + 1
        i = i + 1

    print("Correct Results = {}/{}".format(correctResults, i))
