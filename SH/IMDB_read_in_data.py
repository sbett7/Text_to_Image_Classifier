from os import listdir
from os.path import isfile, join
from SH.HdidxClass import HdidxClass
import numpy as np
import random

db_directory = '..\SH\Data\IMDB\\'

positive_file_location = '..\aclImdb\\train\pos'
negative_file_location = '..\aclImdb\\train\\neg'
positive_files = [f for f in listdir(positive_file_location) if isfile(join(positive_file_location, f))]
num_pos_files = len(positive_files)
file_counter = 0
negative_files = [f for f in listdir(negative_file_location) if isfile(join(negative_file_location, f))]

files = positive_files + negative_files
iRows = [(128, 128), (64, 64), (48, 64), (128, 64)]
nbits_array = [8, 16, 32, 64, 128]
database_size = len(files)
sizeVals = len(files)
balance = int(sizeVals * 0.5)
positiveCounter = 0
negativeCounter = 0
color_image = False
full_image = False
gray_scale = False

IS_COLOR = 0
IS_FULL_IMAGE = 1
IS_GRAYSCALE = 2
PATH = 3

configuration = [[False, False, False, "blackwhite/string_image"],
                 [True, False, False, "color/string_image/not_smooth"],
                 [True, False, True, "color/string_image/smooth"],
                 [True, True, False, "color/full_image/not_smooth"],
                 [True, True, True, "color/full_image/smooth"],
                 [True, False, True, "color/string_image/smooth"],
                 [False, True, True, "grayscale/full_image"],
                 [False, True, False, "blackwhite/full_image"],
                 [False, False, True, "grayscale/string_image"]]


text_data = []
ran = random.sample(range(0, len(files)), len(files))
mid_point = database_size / 2
data = []
for text in files:
    if file_counter < num_pos_files:
        with open('{}/{}'.format(positive_file_location, text),mode='r') as data:
            text_data.append([data.read(), 1])
    else:
        with open('{}/{}'.format(negative_file_location, text), mode='r') as data:
            text_data.append([data.read(), 0])
    file_counter = file_counter + 1

text = []
for values in text_data:
    text.append(values[0])

# for every values
for config in configuration:
    isColor = config[IS_COLOR]
    isFull = config[IS_FULL_IMAGE]
    isGray = config[IS_GRAYSCALE]
    path = config[PATH]
    for iRow in iRows:
        for nbit in nbits_array:
            print("Config: {}       iRow: {}     nBit: {}".format(path, iRow, nbit))
            image_rows = iRow
            nbits = nbit

            hdidx = HdidxClass(text, 100, image_shape=image_rows, hash_size=nbits,
                               gray_scale=isGray, full_images=isFull, color_images=isColor)
            database = open('{}/{}bit/({}, {})/{}/database.csv'.format(db_directory, nbit, iRow[0], iRow[1], path),
                            mode='w')

            for value in ran:
                data = text_data[value]
                query = (hdidx.get_query_code(data[0], 0, image_shape=image_rows,
                                              gray_scale=isGray, full_image=isFull, color_image=isColor))[0]
                print(query)
                database.write(np.array2string(query) + ", {}\n".format(int(data[1])))

