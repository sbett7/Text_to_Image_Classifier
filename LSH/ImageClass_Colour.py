import numpy as np
import csv
import cv2 as cv


class ImageClass:
    _text = ""
    _image_shape = 1
    data = np.zeros((1, 1), dtype=int)
    sentiment = 0
    id = 0
    _full_image = False
    _is_colour = False
    _gray_scale = False

    # CONSTANTS
    _ROW = 0
    _COLUMN = 1

    ID = 0
    SENTIMENT = 1
    TEXT = 2

    _HEX = 16
    _TWO_CHARACTERS = 2

    def __init__(self, input_text, id_val=0, sentiment=0, image_shape=(32, 70), full_image=False,
                 is_colour=False, gray_scale=False):
        '''
        Converts a set of text to an image based upon the given set of image configurations.
        :param input_text:
        :param id_val: the associated ID for the text if it were to come from a set of text.
        This value is not required.
        :param image_shape: An array specifying the width and height of the image to generate.
        :param full_image: A boolean specifying whether the text will be set to fill the entire image (True) or the
         image will only contain a single iteration of the text string (False).
        :param is_colour: A boolean specifiying whether the generated image will be an RGB Image (True), or a Binary
        Image (False).
        :param gray_scale: A boolean that specifies whether an image will be generated as gray_scale or smoothed (True),
        or if the image will be a binary image. If color_images is True, the resulting image will be smoothed.
        '''
        self._full_image = full_image
        self._is_colour = is_colour
        self._gray_scale = gray_scale
        self._image_shape = image_shape
        self._text = input_text
        self.id = id_val
        self.sentiment = sentiment

        # convert text to image
        self.initialise_array()
        self.convert_to_image()

    def convert_to_image(self):
        '''
        Converts stored text to an image based on the image configurations.
        :return: None
        '''
        if self._full_image:
            if self._is_colour:
                self.convert_string_to_colour_full_image()
            elif self._gray_scale:
                self.convert_string_to_binary_full_image()
                self.data = self.smooth_image()
            else:
                self.convert_string_to_binary_full_image()
        else:
            if self._is_colour:
                self.convert_string_to_colour_image()
            elif self._gray_scale:
                self.convert_string_to_binary()
                self.data = self.smooth_image()
            else:
                self.convert_string_to_binary()

    def convert_string_to_binary_full_image(self):
        '''
        converts the text string to a binary image by converting the text to a binary string and assigning the pixel
        value to 255 or 0 based upon the corresponding binary string character.
        The binary string will be iterated through until the image has been filled.
        :return: None
        '''
        # get binary string
        binary_string = ''.join(format(ord(x), 'b') for x in self._text.strip())

        counter = 0
        row = 0
        column = -1

        # until every element in the image array has been iterated over
        while counter < self.data.shape[self._ROW] * self.data.shape[self._COLUMN]:
            for character in binary_string:
                # increment  column and counter iterators
                column = column + 1
                counter = counter + 1

                # if the number of columns in the image matches the column iterator, move to the next row
                if column == self.data.shape[self._COLUMN]:
                    column = 0
                    row = row + 1
                # if the index has reached the end of the image, break from loop
                if row == (self.data.shape[self._ROW]):
                    break

                # set image value
                if character == '1':
                    self.data[row][column] = 255
                else:
                    self.data[row][column] = 0

    def convert_string_to_binary(self):
        '''
        converts the text string to a binary image by converting the text to a binary string and assigning the pixel
        value to 255 or 0 based upon the corresponding binary string character.
        The binary string will be iterated through once, all pixels in the image that have not been iterated over will
        remain as their initial value.
        :return: None
        '''
        # get binary string
        binary_string = ''.join(format(ord(x), 'b') for x in self._text.strip())

        row = 0
        column = -1
        for character in binary_string:
            # increment  column iterator
            column = column + 1

            # if the number of columns in the image matches the column iterator, move to the next row
            if column == self.data.shape[self._COLUMN]:
                column = 0
                row = row + 1

            # if the index has reached the end of the image, break from loop
            if row == (self.data.shape[self._ROW]):
                break

            # set image value
            if character == '1':
                self.data[row][column] = 255
            else:
                self.data[row][column] = 0

    def convert_string_to_colour_full_image(self, magnitude=2):
        '''
        converts the text string to an RGB image by converting the text to a hexadecimal string and assigning each
        hexadecimal value to a colour element of a pixel.
        The hexadecimal string will be iterated through until the image has been filled.
        :param magnitude: A scaling integer to multiply the value for each pixel.  This is used to make it easy to view
        the pixel values in the generated image.
        :return: None
        '''
        # get hex string
        hex_string = self._text.strip().encode('hex')

        row = 0
        column = -1
        rgb_next = 0
        RED = 0
        BLUE = 1
        GREEN = 2
        red = 0
        blue = 0
        green = 0
        counter = 0
        characters = ""  # variable to store hex characters

        # until every element in the image array has been iterated over
        while counter < self.data.shape[self._ROW] * self.data.shape[self._COLUMN]:
            for character in hex_string:
                characters = characters + character
                # if there are two characters in characters variable, add to image and reset variable
                if len(characters) == self._TWO_CHARACTERS:
                    if rgb_next == RED:
                        rgb_next = BLUE
                        red = int(characters, self._HEX)
                    elif rgb_next == BLUE:
                        rgb_next = GREEN
                        blue = int(characters, self._HEX)
                    else:  # reset colour variable to RED, and set the values of the image element
                        rgb_next = RED
                        green = int(characters, self._HEX)
                        column = column + 1
                        counter = counter + 1

                        # if the number of columns in the image matches the column iterator, move to the next row
                        if column == self.data.shape[self._COLUMN]:
                            column = 0
                            row = row + 1
                        # if the index has reached the end of the image, break from loop
                        if row == (self.data.shape[self._ROW]):
                            break

                        # set values and multiply by the given magnitude.
                        self.data[row][column][RED] = red * magnitude
                        self.data[row][column][BLUE] = blue * magnitude
                        self.data[row][column][GREEN] = green * magnitude
                    characters = ""

    def convert_string_to_colour_image(self, magnitude=2):
        '''
        converts the text string to an RGB image by converting the text to a hexadecimal string and assigning each
        hexadecimal value to a colour element of a pixel.
        The hexadecimal string will be iterated through once, all pixels in the image that have not been iterated
        over will remain as their initial value.
        :param magnitude: A scaling integer to multiply the value for each pixel.  This is used to make it easy to view
        the pixel values in the generated image.
        :return: None
        '''
        # get hex string
        hex_string = self._text.strip().encode('hex')

        row = 0
        column = -1
        rgb_next = 0
        RED = 0
        BLUE = 1
        GREEN = 2
        red = 0
        blue = 0
        green = 0

        characters = ""  # variable to store hex characters
        for character in hex_string:
            characters = characters + character

            # if there are two characters in characters variable, add to image and reset variable
            if len(characters) == self._TWO_CHARACTERS:
                if rgb_next == RED:
                    rgb_next = BLUE
                    red = int(characters, self._HEX)
                elif rgb_next == BLUE:
                    rgb_next = GREEN
                    blue = int(characters, self._HEX)
                else:  # reset colour variable to RED, and set the values of the image element
                    rgb_next = RED
                    green = int(characters, self._HEX)
                    column = column + 1
                    if column == self.data.shape[self._COLUMN]:
                        column = 0
                        row = row + 1

                    # if the index has reached the end of the image, break from loop
                    if row == (self.data.shape[self._ROW]):
                        break

                    # set values and multiply by the given magnitude.
                    self.data[row][column][RED] = red * magnitude
                    self.data[row][column][BLUE] = blue * magnitude
                    self.data[row][column][GREEN] = green * magnitude
                characters = ""

    def initialise_array(self):
        '''
        Initialises the image array based upon whether the image is a RGB or binary image
        :return: None
        '''
        if self._is_colour:
            self.data = np.zeros((self._image_shape[0], self._image_shape[1], 3))
        else:
            self.data = np.zeros((self._image_shape[0], self._image_shape[1]))

    def get_image_vector(self):
        '''
        Gets the image vector of the generated image and returns it.  The image vector is generated depending upon the
        type of image.
        :return: An image vector representation of the generated image.
        '''
        if self._gray_scale:
            self.data = self.smooth_image()

        if self._is_colour:
            img_vector = self._get_colour_image_vector()
        else:
            img_vector = self._get_binary_image_vector()

        return img_vector

    def _get_colour_image_vector(self):
        '''
        Converts the RGB image to an image vector.  This is done by averaging the RGB values of each element, and then
        averaging each column to generate the vector.
        :return: A 1-D image vector representation of the generated image.
        '''
        img_vector = np.zeros((1, self._image_shape[self._COLUMN]), dtype=float)
        gray_scale = np.zeros((self._image_shape[self._ROW], self._image_shape[self._COLUMN]), dtype=float)

        # get average RGB color across every pixel in image
        for i in range(0, self._image_shape[self._ROW] - 1):
            for j in range(0, self._image_shape[self._COLUMN] - 1):
                gray_scale[i, j] = np.mean(self.data[i, j])

        # get average values of each column
        for i in range(0, self._image_shape[self._COLUMN] - 1):
            img_vector[self._ROW][i] = np.mean(gray_scale[:, i])

        return img_vector

    def _get_binary_image_vector(self):
        '''
        Converts the binary image to an image vector.  This is done by averaging each column to generate the vector.
        :return: A 1-D image vector representation of the generated image.
        '''

        img_vector = np.zeros((1, self._image_shape[1]), dtype=float)

        # get average values of each column
        for i in range(0, self._image_shape[1] - 1):
            img_vector[0][i] = np.mean(self.data[:, i])
        return img_vector

    def smooth_image(self):
        '''
        Smooths the image using a 5x5 window.
        :return: The smoothed image
        '''
        kernel = np.ones((5, 5), np.float32) / 25
        return cv.filter2D(self.data, -1, kernel)

    def write_to_image(self, image_name):
        '''
        Writes the image data to an image with the specified name.  The location of the images will be stored in the
        Images folder in the current directory.
        :param image_name:  A string specifying the name of the image.
        :return: None
        '''
        cv.imwrite('Images/{}.jpg'.format(image_name), self.data)

    @staticmethod
    def convert_text_to_image(path_to_store, filename, image_shape=10, is_color=False,
                              full_image=False, gray_scale=False):
        '''
        Converts a set of text to images and writes the images to the specified location.
        :param path_to_store: A string specifying the location for the images to be stored.
        :param filename: A string specifying the name of the image.
        :param image_shape: An array specifying the width and height of the image to generate.
        :param is_color: A boolean specifiying whether the generated image will be an RGB Image (True), or a Binary
        Image (False).
        :param full_image: A boolean specifying whether the text will be set to fill the entire image (True) or the
         image will only contain a single iteration of the text string (False).
        :param gray_scale: A boolean that specifies whether an image will be generated as gray_scale or smoothed (True),
        or if the image will be a binary image. If color_images is True, the resulting image will be smoothed.
        :return:
        '''

        # open file and read in data
        with open(filename, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                if row[0] != "ItemID":
                    item = ImageClass(row[2], int(row[0]),
                                      sentiment=row[1], image_shape=image_shape,
                                      is_colour=is_color, full_image=full_image, gray_scale=gray_scale)
                    cv.imwrite(path_to_store + "TrainImage" + str(item.id) + '.jpg', item.data)
