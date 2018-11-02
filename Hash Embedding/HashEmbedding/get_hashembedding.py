import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from layers_spectral import HashEmbedding, ReduceSum
from keras.layers import Input, Dense, Activation, Embedding
from keras.models import Model
import hashlib
import keras
import numpy as np
from keras.callbacks import EarlyStopping
import csv
import time



def split_string(is_lsh, data_val):
    if is_lsh:
        return str.split(data_val)[0]
    else:
        return str.split(data_val)

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
    return (v % (max_idx - 1)) + 1


if __name__ == '__main__':

    db_path = '..\Data\IMDB'
    results_path = '..\Data\IMDB\results_hashembedding.csv'
    is_LSH = False

    iRows_data = [(28, 80),(40, 28),(35, 32)]
    nbits_array = [8, 16, 32, 64, 128]

    configuration = [[False, False, False, "blackwhite/string_image"],
                     [True, False, True, "color/string_image/not_smooth"],
                     [True, True, False, "color/full_image/not_smooth"],
                     [False, True, True, "grayscale/full_image"],
                     [False, True, False, "blackwhite/full_image"],
                     [False, False, True, "grayscale/string_image"]]

    with open(results_path, mode='w') as result_data_file:
        field_names = ['Hash Size', 'Image Shape', 'Image Type', 'Accuracy', 'Fit Time',
                       'Average Query Time', 'acc', 'loss']
        writer = csv.DictWriter(result_data_file, fieldnames=field_names)
        writer.writeheader()
        for nbit in nbits_array:
            for iRows in iRows_data:
                for config in configuration:
                    path = config[3]
                    data_file = '{}/{}bit/({}, {})/{}/database.csv'.format(db_path, nbit, iRows[0], iRows[1], path)
                    with open(data_file, 'rt') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                        label_np = np.zeros((1, 2))
                        len_of_data = len(list(reader))
                        labels = np.zeros((len_of_data, 2))
                        data_spectral = [[] for i in range(len_of_data)]

                    with open(data_file, 'rt') as csvfile:
                        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                        i = 0
                        for row in reader:
                            dat = row[0]
                            dat = dat[1:len(dat) - 1]
                            values = split_string(is_LSH, dat)
                            data_spectral[i] = []
                            for val in values:
                                data_spectral[i].append(int(val))

                            label_np[0, 0] = int(row[1])
                            if label_np[0, 0] == 1:
                                labels[i, 0] = 1
                                labels[i, 1] = 0
                            else:
                                labels[i, 0] = 0
                                labels[i, 1] = 1
                            i = i + 1

                    use_hash_embeddings = True
                    embedding_size = nbit
                    num_buckets = 5000  # number of buckets in second hashing layer (hash embedding)
                    max_words = 10 ** 2  # number of buckets in first hashing layer
                    max_epochs = 50
                    num_hash_functions = 1
                    num_classes = 2

                    if use_hash_embeddings:
                        embedding = HashEmbedding(max_words, num_buckets, embedding_size,
                                                  num_hash_functions=num_hash_functions)
                    else:
                        embedding = Embedding(max_words, embedding_size)

                    idx_test = int(len(data_spectral) * 0.75)

                    data_encoded = np.vstack(data_spectral)

                    model = get_model(embedding, num_classes)
                    metrics = ['accuracy']
                    loss = 'sparse_categorical_crossentropy'
                    model.compile(optimizer=keras.optimizers.Adam(),
                                  loss=loss, metrics=['accuracy'])

                    print('Num parameters in model: %i' % model.count_params())
                    fit_start = time.time()
                    model.fit(data_encoded[0:idx_test], labels[0:idx_test, 1], validation_split=0.1,
                              nb_epoch=max_epochs, callbacks=[EarlyStopping(patience=10)])
                    fit_time = time.time() - fit_start

                    test_result = model.test_on_batch(data_encoded[idx_test:len_of_data],
                                                      labels[idx_test:len_of_data, 1])

                    accuracies = [0, 0]
                    iterator = 0
                    for i, (name, res) in enumerate(zip(model.metrics_names, test_result)):
                        accuracies[iterator] = ('%1.4f' % res)
                        print('%s: %1.4f' % (name, res))
                        iterator = iterator + 1

                    query_start = time.time()
                    results = model.predict(data_encoded[idx_test:len_of_data])
                    query_time = float(time.time() - query_start) / len(results)

                    start = idx_test
                    correctResults = 0
                    i = 0
                    for result in results:
                        if labels[start + i, 0] == 1:
                            correctResults = correctResults + 1
                        i = i + 1

                    print("Correct Results = {}/{}".format(correctResults, len(results)))

                    accuracy = float(correctResults) / len(results)

                    writer.writerow({
                        'Hash Size': nbit,
                        'Image Shape': '({}, {})'.format(iRows[0], iRows[1]),
                        'Image Type': path,
                        'Accuracy': accuracy,
                        'Fit Time': fit_time,
                        'Average Query Time': query_time,
                        'acc': accuracies[1],
                        'loss': accuracies[0]
                    })

                    print('Hash Size: {}, Image Shape: ({}, {}), Image Type: {},'
                          ' Accuracy: {}, Fit Time: {}, Average Query Time: {}'.format(nbit, iRows[0], iRows[1],
                                                                                       path,
                                                                                       accuracy, fit_time,
                                                                                       query_time))