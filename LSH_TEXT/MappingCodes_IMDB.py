from os import listdir
from os.path import isfile, join
import LshClass
import numpy as np
from datasketch import MinHash

ROWS = 0
NBITS = 1
IMAGE_ROWS = 2
NUM_QUERIES = 3

TEXT = 2
SENTIMENT = 1

positive_file_location = '/media/sean/My Passport/Data/aclImdb_v1(1)/aclImdb/train/pos/'
negative_file_location = '/media/sean/My Passport/Data/aclImdb_v1(1)/aclImdb/train/neg/'
positive_files = [f for f in listdir(positive_file_location) if isfile(join(positive_file_location, f))]
num_pos_files = len(positive_files)
file_counter = 0
negative_files = [f for f in listdir(negative_file_location) if isfile(join(negative_file_location, f))]

files = positive_files + negative_files
database_size = len(files)
text_data = []
ran = range(1, database_size)
mid_point = database_size / 2

num_perms = [2, 4, 8, 16, 32, 64, 128, 256, 512]


def get_codes(text_val, nbit=128):
    '''
    Converts the text data to a binary code.
    :param text_val: The text data to be converted to a minhash binary code.
    :param nbit: The size of the binary code to be returned.
    :return: A binary code representation of the text string.
    '''
    minhash = MinHash(num_perm=nbit)
    for c, i in enumerate(text_val):
        minhash.update("".join(i))
    return minhash


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

for num_perm in num_perms:
    database = open('/media/sean/My Passport/Data/Hashing/IMDB/perm{}/database.txt'.format(num_perm), mode='w')

    for i in ran:
        data = text_data[i]
        query = get_codes(data[0], nbit=num_perm)

        queryDat = np.zeros(len(query.hashvalues), dtype=int)
        queryString = None
        queryList = []
        for value in query.hashvalues.astype(int):
            string = str(value) + ' '
            queryList.append(string)
        queryString = ''.join(queryList)
        print(queryString)
        try:
            if int(data[1]) == 1:
                database.write("[" + queryString + "], 1" + ", 0" + "\n")
            else:
                database.write("[" + queryString + "], 0" + ", 1" + "\n")
        except AttributeError:
            print("error")

