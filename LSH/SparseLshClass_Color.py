from ImageClass_Colour import ImageClass as Ic
import numpy as np
from sparselsh import LSH
from scipy.sparse import csr_matrix
import csv


class SparseLshClass:

    ID = 0
    SENTIMENT = 1
    TEXT = 2
    _text_image_list = []
    _query_val = []
    _processed_data = np.zeros((1, 1))
    _query_data = np.zeros((1, 1))
    _length_of_data = 0
    _data_csr_matrix = csr_matrix(0)
    _lsh = []

    # IMAGE CREATION PROPERTIES
    _full_images = False
    _gray_scale = False
    _color_images = False
    _image_shape = np.zeros((1, 1))

    textData = []

    def __init__(self, text_data, num_text_entries, image_shape=(64, 64), hash_size=64, num_hashtables=1,
                 full_images=False, gray_scale=False, color_images=False):
        '''
        Creates a Sparse LSH handler class to create, query, and generate binary codes from a SparseLSH object.
        This class can be used to perform manual querying of the constructed SparseLSH indexer, or can be used to
        retrieve binary codes from a given set of _text.  It can also be used as a Sentiment Classifier based off of the
        averaged sentiment returned from the SparseLSH Indexer.
        :param text_data: A list of string with the
        :param num_text_entries: The number of entries that are to be stored in the SparseLSH Indexer
        :param image_shape: The shape of the images that are
        :param hash_size: The size of the binary code that is generated by the SparseLSH Indexer
        :param num_hashtables: The number of Hash Tables that are used in SparseLSH Indexer
        :param full_images: A boolean that specifies whether images will fill the image with a sample
         _text until it has been completely filled.
        :param gray_scale: A boolean that specifies whether an image will be generated as gray_scale or smoothed.
        If color_images is True, the resulting image will be smoothed
        :param color_images: A boolean to specify whether the generated image will be RGB or Binary.
        If True, the image will be an RGB image.
        '''
        # initialise settings
        self.textData = text_data
        self._length_of_data = num_text_entries
        self._full_images = full_images
        self._gray_scale = gray_scale
        self._color_images = color_images
        self._image_shape = image_shape

        # process _processed_data into _processed_data array
        self.get_data(self.textData, num_data_rows=num_text_entries)

        # initialise Numpy Data array and configure LSH Matrix
        self.initialise_data_array()
        self.configure_lsh_matrix(hash_size=hash_size, num_hashtables=num_hashtables)

    def get_data(self, text_data, num_data_rows=5000):
        '''
        Store _text _processed_data as ImageClass object
        :param text_data: The _text _processed_data that is to be added to the Indexer.
        :param num_data_rows: The number of entries that are to be added to the Indexer.
        :return: None
        '''

        for i in range(0, num_data_rows - 1):
            self._text_image_list.append(Ic(text_data[i], 0, image_shape=self._image_shape,
                                            is_colour=self._color_images, gray_scale=self._gray_scale,
                                            full_image=self._full_images))

    def get_data_row(self, index):
        '''
        Get a specific _text string  based upon the provided index.
        :param index: An index for the requested set of _text.
        :return: a _text string for the specified index.
        '''
        return self.textData[index]

    def initialise_data_array(self):
        '''
        Intialises the data array that will be fed to the SparseLSH Index.
        :return: None
        '''
        # initialise processed data array
        self._processed_data = np.zeros((len(self._text_image_list), self._image_shape[1]))
        i = 0
        for row in self._text_image_list:
            dat = row.get_image_vector()

            for j in range(0, row.arrayRows[1]):
                self._processed_data[i][j] = dat[0][j]
            i = i + 1

    def initialise_query_array(self):
        '''
        Intialises the query array that will be used to query the SparseLSH Index.
        :return:
        '''
        # initialise query data array
        self._query_data = np.zeros((1, self._image_shape[1]))
        dat = self._query_val.get_image_vector()

        for j in range(0, self._query_val._image_shape[1]):
            self._query_data[0][j] = dat[0][j]

    def query(self, text, num_queries, id_val, image_shape, full_image=False,
              gray_scale=False, color_image=False):
        '''
        Queries the SparseLSH index with the specified _text and image configurations.
        :param text: The _text string that is to be used to query the SparseLSH Index.
        :param num_queries: The number of results returned by the query.
        :param id_val: The ID for the _text.
        :param image_shape: The shape of the generated Image
        :param full_image: A boolean that specifies whether images will fill the image with a sample
         _text until it has been completely filled.
        :param gray_scale: A boolean that specifies whether an image will be generated as gray_scale or smoothed.
        If color_images is True, the resulting image will be smoothed
        :param color_image: A boolean to specify whether the generated image will be RGB or Binary.
        If True, the image will be an RGB image.
        :return: The results vector from the SparseLSH Index.
        '''

        # intialise query object
        self._query_val = Ic(text, id_val=id_val, image_shape=image_shape, full_image=full_image,
                             gray_scale=gray_scale, is_colour=color_image)
        self.initialise_query_array()

        # query the Index and return the results
        results = self._lsh.query(csr_matrix(self._query_data), num_results=num_queries)
        return results

    def configure_lsh_matrix(self, hash_size=4, num_hashtables=1):
        '''
        Configures the SparseLSH Indexer object with the hash sizes and hash tables.
        :param hash_size: The size of the binary codes that will be returned from the LSH hashing function.
        :param num_hashtables: The number of hash tables that will be used to perform index search querying.
        :return: None
        '''
        self._data_csr_matrix = csr_matrix(self._processed_data)
        self._lsh = LSH(hash_size, self._data_csr_matrix.shape[1],
                        num_hashtables=num_hashtables,
                        storage_config={"dict": None}
                        )
        # get list of sentiment to index with data for querying
        sentiment_list = []
        for row in self._text_image_list:
            sentiment_list.append(row.sentiment)

        # build index
        for ix in range(self._data_csr_matrix.shape[0]):
            data_vals = self._data_csr_matrix.getrow(ix)
            sentiment_vals = sentiment_list[ix]
            self._lsh.index(data_vals, extra_data=sentiment_vals)

    def configure_query_matrix(self):
        '''
        Configures the query matrix into a form that can be fed into the Index.
        :return: SciPy CSR_Matrix object of the query data.
        '''
        return csr_matrix(self._query_data)

    def get_sentiment_of_text(self, text, num_returned_results=5, text_id=0, image_shape=10, threshold=0.5,
                              full_image=False, gray_scale=False, color_image=False):
        '''
        Determines the sentiment of given _text by determining the average sentiment from the queried results.
        :param text: A string of _text that is to be used to query the Index to determine its sentiment.
        :param num_returned_results: The number of results that a query returns.
        :param text_id: The ID of a given set of _text.  This is used if their is a specific ID associated with the _text
        data.
        :param image_shape: The shape of the image that the _text will be converted to.
        :param threshold: The threshold that will be used to determine if the sentiment of the _text is positive or
         negative.  If the average sentiment of the returned results is above or equal to the threshold, it will return
         a positive sentiment (True), otherwise it will return a negative sentiment (False).
        :param full_image: A boolean that specifies whether images will fill the image with a sample
         _text until it has been completely filled.
        :param gray_scale: A boolean that specifies whether an image will be generated as gray_scale or smoothed.
        If color_images is True, the resulting image will be smoothed
        :param color_image: A boolean to specify whether the generated image will be RGB or Binary.
        If True, the image will be an RGB image.
        :return: A boolean to indicate whether the sentiment is positive or negative.  If True, the sentiment of the
        _text is positive. If False, the sentiment of the _text is negative.
        '''
        results = self.query(text, num_returned_results, text_id, image_shape, full_image=full_image,
                            gray_scale=gray_scale, color_image=color_image)

        # calculates the sentiment from the average number of queries
        average = float(self.get_average_sentiment(results, num_results=num_returned_results))
        if average >= threshold:
            return True
        else:
            return False

    def get_query_code(self, text, image_shape, id_val=0,
                       full_image=False, gray_scale=False, color_image=False):
        '''
        Gets the binary code representation of a given _text string that has been converted into an image based upon the
        given image configuration.
        :param text: A string of _text that is to be used to get a binary code based on the image
         generated from the _text.
        :param image_shape: The shape of the image that the _text will be converted to.
        :param id_val: The ID for the _text.
        :param full_image: A boolean that specifies whether images will fill the image with a sample
         _text until it has been completely filled.
        :param gray_scale: A boolean that specifies whether an image will be generated as gray_scale or smoothed.
        If color_images is True, the resulting image will be smoothed
        :param color_image: A boolean to specify whether the generated image will be RGB or Binary.
        If True, the image will be an RGB image.
        :return: The binary code representation of the provided _text that has been converted into an image
        '''
        # convert _text to image
        self._query_val = Ic(text, id_val=id_val, image_shape=image_shape, gray_scale=gray_scale,
                             full_image=full_image, is_colour=color_image)
        # initialise query data structure and perform hash function on query
        self.initialise_query_array()
        query_encode = self._lsh.get_binary_code(self._query_data)
        return query_encode

    @staticmethod
    def get_average_sentiment(results, num_results):
        '''
        Averages the sentiment of all of the returned results.
        :param results: An array of all results that were retrieved from the SparseLSH Index query.
        :param num_results: The number of returned results.
        :return: A float containing the averaged result of the returned sentiments.
        '''
        average = 0
        for i in range(0, num_results):
            average = float(average + results[i][0][1]) / float(num_results)
        return average

    @staticmethod
    def read_in_text_data(file_location):
        '''
        Reads in data from a specified _text file.
        :param file_location: The location of the file to retrieve the data from.
        :return: A list with all of the data entries.
        '''
        text_data = []
        with open(file_location, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                text_data.append(row)
        return text_data
