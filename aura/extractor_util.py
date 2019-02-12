import numpy as np
import scipy.misc


@DeprecationWarning
def convert_to_size(input, size):
    """
    Converts an input image to size
    Just use scipy.misc.imresize... It's much easier and better.

    :param input: 2D Numpy array image
    :param size: The size to be converted to
    :return: A 2D numpy array
    """
    output = np.zeros(size, dtype=np.float16)
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            if row < len(input) and col < len(input[row]):
                output[row][col] = input[row][col]

    return output


def reshape(input, size):
    """
    Reshapes a 3D array of 2D arrays.

    :param input: A 3D numpy array
    :param size: Image sizes to be converted to
    :return: A 3D array with reshaped numpy arrays.
    """

    output = np.zeros(size, dtype=np.float16)
    for image in range(output.shape[2]):
        output[:, :, image] = scipy.misc.imresize(input[:, :, image], size[0:2])

    return output


def parse_aura_dimensions(dimensions):
    """
    Takes an input like {10x10x100} and returns a tuple of 10,10,100

    :param dimensions: A string type of the dimensions of an aura file
    :return: A tuple with aura dimensions.
    """

    l, w, n = dimensions[dimensions.find("{") + 1: dimensions.rfind("}")].split("x")
    l, w, n = int(l), int(w), int(n)
    return l, w, n
