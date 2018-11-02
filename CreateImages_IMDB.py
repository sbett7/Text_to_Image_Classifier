from os import listdir
from os.path import isfile, join
from SH.ImageClass_Colour import ImageClass as Ic
import numpy as np
import cv2 as cv
import random

DATABASE = 0
TEST = 1
TRAIN = 2
save_location = '..\DeepHash-master\data\IMDB'
positive_file_location = '..\aclImdb\train\pos'
negative_file_location = '..\aclImdb\train\neg'
iRows = [(128, 128), (64, 64), (48, 64), (128, 64)]



def write_file(writer, file_name, sentiment):
    if sentiment == 0:
        writer.write('{}.jpg 1 0\n'.format(file_name))
    else:
        writer.write('{}.jpg 0 1\n'.format(file_name))


def write_to_train_test(writers, directory, file_data, file_name,sentiment,image_id, midpoint):
    if image_id <= midpoint:
        write_file(writers[TEST], '{}/test/{}'.format(directory, file_name), sentiment)
        write_file(writers[DATABASE], '{}/test/{}'.format(directory, file_name), sentiment)
        cv.imwrite('{}/test/{}.jpg'.format(directory, file_name),file_data)
    else:
        write_file(writers[TRAIN], '{}/train/{}'.format(directory, file_name), sentiment)
        write_file(writers[DATABASE], '{}/train/{}'.format(directory, file_name), sentiment)
        cv.imwrite('{}/train/{}.jpg'.format(directory, file_name), file_data)


positive_files = [f for f in listdir(positive_file_location) if isfile(join(positive_file_location, f))]
num_pos_files = len(positive_files)
file_counter = 0
negative_files = [f for f in listdir(negative_file_location) if isfile(join(negative_file_location, f))]

files = positive_files + negative_files

text_data = []
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

ran = random.sample(range(0, len(files)), len(files))

color_fullimage_notsmoothed = '{}\color_fullimage_notsmoothed'.format(save_location)
color_fullimage_smoothed = '{}\color_fullimage_smoothed'.format(save_location)
color_stringimage_notsmoothed = '{}\color_stringimage_notsmoothed'.format(save_location)
color_stringimage_smoothed = '{}\color_stringimage_smoothed'.format(save_location)
gray_scale_fullimage = '{}\gray_scale_fullimage'.format(save_location)
gray_scale_stringimage = '{}\gray_scale_stringimage'.format(save_location)
blackwhite_fullimage = '{}\\blackwhite_fullimage'.format(save_location)
blackwhite_stringimage = '{}\\blackwhite_stringimage'.format(save_location)

image_ID = 0
for iRow in iRows:

    color_fullimage_notsmoothed_writers = [open('{}/{}_{}/database.txt'.format(color_fullimage_notsmoothed,
                                                                               iRow[0], iRow[1]),
                                                mode='w'),
                                           open('{}/{}_{}/test.txt'.format(color_fullimage_notsmoothed,
                                                                               iRow[0], iRow[1]),
                                                mode='w'),
                                           open('{}/{}_{}/train.txt'.format(color_fullimage_notsmoothed,
                                                                               iRow[0], iRow[1]),
                                                mode='w')]

    color_fullimage_smoothed_writers = [open('{}/{}_{}/database.txt'.format(color_fullimage_smoothed,
                                                                               iRow[0], iRow[1]),
                                                mode='w'),
                                           open('{}/{}_{}/test.txt'.format(color_fullimage_smoothed,
                                                                               iRow[0], iRow[1]),
                                                mode='w'),
                                           open('{}/{}_{}/train.txt'.format(color_fullimage_smoothed,
                                                                               iRow[0], iRow[1]),
                                                mode='w')]

    color_stringimage_notsmoothed_writers = [open('{}/{}_{}/database.txt'.format(color_stringimage_notsmoothed,
                                                                               iRow[0], iRow[1]),
                                                mode='w'),
                                           open('{}/{}_{}/test.txt'.format(color_stringimage_notsmoothed,
                                                                               iRow[0], iRow[1]),
                                                mode='w'),
                                           open('{}/{}_{}/train.txt'.format(color_stringimage_notsmoothed,
                                                                               iRow[0], iRow[1]),
                                                mode='w')]

    color_stringimage_smoothed_writers = [open('{}/{}_{}/database.txt'.format(color_stringimage_smoothed,
                                                                               iRow[0], iRow[1]),
                                                mode='w'),
                                           open('{}/{}_{}/test.txt'.format(color_stringimage_smoothed,
                                                                               iRow[0], iRow[1]),
                                                mode='w'),
                                           open('{}/{}_{}/train.txt'.format(color_stringimage_smoothed,
                                                                               iRow[0], iRow[1]),
                                                mode='w')]

    gray_scale_fullimage_writers = [open('{}/{}_{}/database.txt'.format(gray_scale_fullimage,
                                                                               iRow[0], iRow[1]),
                                                mode='w'),
                                           open('{}/{}_{}/test.txt'.format(gray_scale_fullimage,
                                                                               iRow[0], iRow[1]),
                                                mode='w'),
                                           open('{}/{}_{}/train.txt'.format(gray_scale_fullimage,
                                                                               iRow[0], iRow[1]),
                                                mode='w')]

    gray_scale_stringimage_writers = [open('{}/{}_{}/database.txt'.format(gray_scale_stringimage,
                                                                               iRow[0], iRow[1]),
                                                mode='w'),
                                           open('{}/{}_{}/test.txt'.format(gray_scale_stringimage,
                                                                               iRow[0], iRow[1]),
                                                mode='w'),
                                           open('{}/{}_{}/train.txt'.format(gray_scale_stringimage,
                                                                               iRow[0], iRow[1]),
                                                mode='w')]

    blackwhite_fullimage_writers = [open('{}/{}_{}/database.txt'.format(blackwhite_fullimage,
                                                                               iRow[0], iRow[1]),
                                                mode='w'),
                                           open('{}/{}_{}/test.txt'.format(blackwhite_fullimage,
                                                                               iRow[0], iRow[1]),
                                                mode='w'),
                                           open('{}/{}_{}/train.txt'.format(blackwhite_fullimage,
                                                                               iRow[0], iRow[1]),
                                                mode='w')]

    blackwhite_stringimage_writers = [open('{}/{}_{}/database.txt'.format(blackwhite_stringimage,
                                                                               iRow[0], iRow[1]),
                                                mode='w'),
                                           open('{}/{}_{}/test.txt'.format(blackwhite_stringimage,
                                                                               iRow[0], iRow[1]),
                                                mode='w'),
                                           open('{}/{}_{}/train.txt'.format(blackwhite_stringimage,
                                                                               iRow[0], iRow[1]),
                                                mode='w')]

    image_ID = 0
    for i in ran:
        print("ItemID: {}".format(image_ID))
        #writer.write("TrainImage" + str(i) + '.jpg, {}\n'.format(text_data[i][1]))
        # Colour Full
        fcFull_not_smooth = Ic(text_data[i][0],image_shape=iRow, is_colour=True, full_image=True, gray_scale=False)
        write_to_train_test(color_fullimage_notsmoothed_writers,"{}/{}_{}".format(color_fullimage_notsmoothed,
                                                                               iRow[0], iRow[1]),
                            fcFull_not_smooth.data, "TrainImage{}".format(i), int(text_data[i][1]), image_ID,
                            len(files)*0.5)

        fcFull_smooth = Ic(text_data[i][0],image_shape=iRow, is_colour=True, full_image=True, gray_scale=True)
        write_to_train_test(color_fullimage_smoothed_writers, "{}/{}_{}".format(color_fullimage_smoothed,
                                                                                      iRow[0], iRow[1]),
                            fcFull_smooth.data, "TrainImage{}".format(i), int(text_data[i][1]), image_ID,
                            len(files) * 0.5)

        # Colour Not Full
        fc_not_smooth = Ic(text_data[i][0],image_shape=iRow, is_colour=True, full_image=False, gray_scale=False)
        write_to_train_test(color_stringimage_notsmoothed_writers, "{}/{}_{}".format(color_stringimage_notsmoothed,
                                                                                      iRow[0], iRow[1]),
                            fc_not_smooth.data, "TrainImage{}".format(i), int(text_data[i][1]), image_ID,
                            len(files) * 0.5)

        fc_smooth = Ic(text_data[i][0],image_shape=iRow, is_colour=True, full_image=False, gray_scale=True)
        write_to_train_test(color_stringimage_smoothed_writers, "{}/{}_{}".format(color_stringimage_smoothed,
                                                                                      iRow[0], iRow[1]),
                            fc_smooth.data, "TrainImage{}".format(i), int(text_data[i][1]), image_ID,
                            len(files) * 0.5)

        # Grayscale
        gray_full = Ic(text_data[i][0],image_shape=iRow, is_colour=False, full_image=True, gray_scale=True)
        write_to_train_test(gray_scale_fullimage_writers, "{}/{}_{}".format(gray_scale_fullimage,
                                                                                      iRow[0], iRow[1]),
                            gray_full.data, "TrainImage{}".format(i), int(text_data[i][1]), image_ID,
                            len(files) * 0.5)

        gray_short = Ic(text_data[i][0],image_shape=iRow, is_colour=False, full_image=False, gray_scale=True)
        write_to_train_test(gray_scale_stringimage_writers, "{}/{}_{}".format(gray_scale_stringimage,
                                                                                      iRow[0], iRow[1]),
                            gray_short.data, "TrainImage{}".format(i), int(text_data[i][1]), image_ID,
                            len(files) * 0.5)

        # BlackWhite
        BW_full = Ic(text_data[i][0],image_shape=iRow, is_colour=False, full_image=True, gray_scale=False)
        write_to_train_test(blackwhite_fullimage_writers, "{}/{}_{}".format(blackwhite_fullimage,
                                                                                      iRow[0], iRow[1]),
                            BW_full.data, "TrainImage{}".format(i), int(text_data[i][1]), image_ID,
                            len(files) * 0.5)

        BW_short = Ic(text_data[i][0],image_shape=iRow, is_colour=False, full_image=False, gray_scale=False)
        write_to_train_test(blackwhite_stringimage_writers, "{}/{}_{}".format(blackwhite_stringimage,
                                                                                      iRow[0], iRow[1]),
                            BW_short.data, "TrainImage{}".format(i), int(text_data[i][1]), image_ID,
                            len(files) * 0.5)

        image_ID = image_ID + 1
    for file_item in color_fullimage_notsmoothed_writers:
        file_item.close()
    for file_item in color_fullimage_smoothed_writers:
        file_item.close()
    for file_item in color_stringimage_notsmoothed_writers:
        file_item.close()
    for file_item in color_stringimage_smoothed_writers:
        file_item.close()
    for file_item in gray_scale_fullimage_writers:
        file_item.close()
    for file_item in gray_scale_stringimage_writers:
        file_item.close()
    for file_item in blackwhite_fullimage_writers:
        file_item.close()
    for file_item in blackwhite_stringimage_writers:
        file_item.close()





