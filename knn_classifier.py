import numpy as np
import csv
from sklearn.neighbors import KNeighborsClassifier
import time


db_path = 'F:\Documents\PycharmProjects\SpectralHashing\LSH\Data\Twitter'  # Path to where data is stored
results_path = 'F:\Documents\PycharmProjects\SpectralHashing\LSH\Data\Twitter\\results_knn.csv'  # Editing data
is_LSH = True
iRows_data = [(28, 80), (35, 32), (40, 28)]
nbits_array = [8, 16, 32, 64, 128]
neighbours_array = [1, 4, 20, 50, 100, 250]
types_array = ['euclidean', 'hamming']
num_data = 0

configuration = [[False, False, False, "blackwhite/string_image"],
                 [True, True, False, "color/full_image/not_smooth"],
                 [False, True, True, "grayscale/full_image"],
                 [False, True, False, "blackwhite/full_image"],
                 [False, False, True, "grayscale/string_image"]]


def split_string(is_lsh, data_val):
    '''
    Splits the data based on whether the data is space separated or not.
    :param is_lsh: A boolean that specifies whether the data being used is LSH.
    :param data_val: The string data that is to be processed.
    :return: Returns the split data string.
    '''
    if is_lsh:
        return str.split(data_val)[0]
    else:
        return str.split(data_val)


def get_length_of_data(data_row):
    '''
    Gets the size of the row data.
    :param data_row: The text sample with the data.
    :return: The number of entries per text sample.
    '''
    dat_val = data_row[0]
    dat_val = dat_val[1:len(dat_val) - 1]
    value = split_string(is_LSH, dat_val)
    value_array = []
    for vals in value:
        value_array.append(int(vals))
    return len(value_array)


def process_data(rows, data_len, data_width):
    '''
    Converts all of the row data to a format that can be passed to classifier.
    :param rows: a list of rows with the data to process.
    :param data_len: The number of rows to be processed.
    :param data_width: The number of entries per entry.
    :return: The converted data array, and the label array.
    '''
    i = 0
    j = 0
    data_array = np.zeros((data_len, data_width), dtype=float)
    label_array = np.zeros((data_len, 2))
    label_temp = 0
    for row in rows:
        dat = row[0]
        dat = dat[1:len(dat) - 1]
        values = split_string(is_LSH, dat)
        j = 0
        for val in values:
            data_array[i, j] = val
            j = j + 1
        try:
            label_temp = int(row[1])
        except ValueError:
            label_temp = int(row[1][0])
        if label_temp == 1:
            label_array[i, 0] = 1
            label_array[i, 1] = 0
        else:
            label_array[i, 0] = 0
            label_array[i, 1] = 1
        i = i + 1
    return data_array, label_array

with open(results_path, mode='w') as result_data_file:
    field_names = ['Hash Size', 'Image Shape', 'Image Type', 'KNN Type', 'Neighbours', 'Accuracy', 'Fit Time',
                                               'Average Query Time']
    writer = csv.DictWriter(result_data_file, fieldnames=field_names)
    writer.writeheader()
    for nbit in nbits_array:
        for iRows in iRows_data:
            for config in configuration:
                path = config[3]
                data_file = '{}/{}bit/({}, {})/{}/database.csv'.format(db_path, nbit, iRows[0], iRows[1], path)

                with open(data_file, 'rt') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                    len_of_data = len(list(reader))

                with open(data_file, 'rt') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                    for row in reader:
                        num_data = get_length_of_data(row)
                        break

                with open(data_file, 'rt') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                    data, labels = process_data(reader, len_of_data, num_data)

                # get data samples
                np.random.seed(42)
                indices = np.random.permutation(len(data))
                n_training_samples = int(data.shape[0] * 0.5)
                learnset_data = data[indices[:-n_training_samples]]
                learnset_labels = labels[indices[:-n_training_samples], 0]
                testset_data = data[indices[-n_training_samples:]]
                testset_labels = labels[indices[-n_training_samples:], 0]

                for neighbour in neighbours_array:
                    for types in types_array:
                        # Create and fit a nearest-neighbor classifier
                        knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric=types,
                                                   metric_params=None,
                                                   n_jobs=1, n_neighbors=neighbour, weights='uniform')

                        # fit data and get time taken to fit
                        fit_start = time.time()
                        knn.fit(learnset_data, learnset_labels)
                        fit_time = time.time() - fit_start

                        # predict data and get query time
                        query_start = time.time()
                        results = knn.predict(testset_data)
                        query_end = time.time()
                        query_time = float(query_end - query_start) / float(len(testset_labels))

                        correctResults = 0
                        i = 0
                        for result in results:
                            if result == testset_labels[i]:
                                correctResults = correctResults + 1
                            i = i + 1
                        accuracy = float(correctResults) / float(len(testset_labels))

                        writer.writerow({
                            'Hash Size': nbit,
                            'Image Shape': '({}, {})'.format(iRows[0], iRows[1]),
                            'Image Type': path,
                            'KNN Type': types,
                            'Neighbours': neighbour,
                            'Accuracy': accuracy,
                            'Fit Time': fit_time,
                            'Average Query Time': query_time
                        })

                        print('Hash Size: {}, Image Shape: ({}, {}), Image Type: {}, KNN Type: {}, Neighbours: {},'
                              ' Accuracy: {}, Fit Time: {}, Average Query Time: {}'.format(nbit, iRows[0], iRows[1],
                                                                                           path, types, neighbour,
                                                                                           accuracy, fit_time,
                                                                                           query_time))

