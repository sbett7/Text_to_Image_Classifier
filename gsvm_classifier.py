from sklearn import svm
from sklearn.model_selection import train_test_split
import time
from sklearn import metrics
import csv
import numpy as np
from scipy.spatial.distance import cdist

db_path = 'F:\Documents\PycharmProjects\SpectralHashing\SH\Data\Twitter'  # Path to where data is stored
results_path = 'F:\Documents\PycharmProjects\SpectralHashing\SH\Data\Twitter\\results_svm.csv'  # Editing data
is_LSH = False  # If using LSH data.

iRows_data = [(28, 80), (35, 32), (40, 28)]
nbits_array = [8, 16, 32, 64, 128]

configuration = [[False, False, False, "blackwhite/string_image"],
                 [True, False, True, "color/string_image/not_smooth"],
                 [True, True, False, "color/full_image/not_smooth"],
                 [False, True, True, "grayscale/full_image"],
                 [False, True, False, "blackwhite/full_image"],
                 [False, False, True, "grayscale/string_image"]]

field_names = ['Hash Size', 'Image Shape', 'Image Type', 'C-Val', 'Accuracy', 'Precision', 'Recall', 'Fit Time',
                                               'Average Query Time']


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


with open(results_path, mode='w') as result_data_file:
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

                # get Gaussian kernel
                Xx_w2v = np.asarray(data_spectral)
                D_w2v = cdist(Xx_w2v, Xx_w2v, 'hamming')
                c_w2v = np.concatenate(D_w2v)  # vectorise it
                kernel_g_w2v = np.median(c_w2v)

                X_train, X_test, Y_train, Y_test = train_test_split(np.asarray(data_spectral),
                                                                    np.asarray(labels[:, 0]),
                                                                    test_size=0.3)  # 70% training and 30% test

                for cVal in [1e-3, 1e-2, 1e-1, 1e+1, 1e+2, 1e+3]:
                    if kernel_g_w2v == 0.0:
                        kernel_g_w2v = 1.0
                    GSVM_w2v_model = svm.SVC(C=cVal, kernel='rbf', gamma=kernel_g_w2v)

                    clf = svm.LinearSVC(C=cVal) # Linear Kernel

                    # fit data and determine time
                    fit_start = time.time()
                    GSVM_w2v_model.fit(X_train, Y_train)
                    fit_time = time.time() - fit_start

                    # Predict the response for test dataset
                    query_start = time.time()
                    GSVM_w2v_predict = GSVM_w2v_model.predict(X_test)
                    query_time = float(time.time() - query_start) / len(X_test)

                    accuracy = metrics.accuracy_score(Y_test, GSVM_w2v_predict)
                    precision = metrics.precision_score(Y_test, GSVM_w2v_predict)
                    recall = metrics.recall_score(Y_test, GSVM_w2v_predict)

                    # write results
                    writer.writerow({
                        'Hash Size': nbit,
                        'Image Shape': '({}, {})'.format(iRows[0], iRows[1]),
                        'Image Type': path,
                        'C-Val': cVal,
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'Fit Time': fit_time,
                        'Average Query Time': query_time
                    })

                    print('Hash Size: {}, Image Shape: ({}, {}), Image Type: {}, C-Val: {},'
                          ' Accuracy: {}, Fit Time: {}, Average Query Time: {}'.format(nbit, iRows[0], iRows[1],
                                                                                       path, cVal,
                                                                                       accuracy, fit_time,
                                                                                       query_time))
