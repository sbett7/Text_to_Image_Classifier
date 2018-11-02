
from SH.HdidxClass import HdidxClass
import numpy as np
import random
import csv

db_directory = '..\SH\Data\Twitter'
TEXT = 2
SENTIMENT = 1
has_header = True

db_file = '..\SH\train.csv'

iRows = [(28, 80),(40, 28),(35, 32)]
nbits_array = [64, 128]

IS_COLOR = 0
IS_FULL_IMAGE = 1
IS_GRAYSCALE = 2
PATH = 3

configuration = [[False, False, False, "blackwhite/string_image"],
                 [True, False, True, "color/string_image/not_smooth"],
                 [True, True, False, "color/full_image/not_smooth"],
                 [False, True, True, "grayscale/full_image"],
                 [False, True, False, "blackwhite/full_image"],
                 [False, False, True, "grayscale/string_image"]]


text_data = []
header = True
with open(db_file, mode='r') as dat_file:
    reader = csv.reader(dat_file, delimiter=',', quotechar='|')
    for row in reader:
        if has_header and header:
            header = False
        else:
            text_data.append(row)

ran = random.sample(range(0, len(text_data)), len(text_data))
mid_point = len(text_data) / 2

text = []
for values in text_data:
    text.append(values[TEXT])

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
                query = (hdidx.get_query_code(data[TEXT], 0, image_shape=image_rows,
                                              gray_scale=isGray, full_image=isFull, color_image=isColor))[0]
                print(query)
                database.write(np.array2string(query) + ", {}\n".format(int(data[1])))

