import scipy.misc
from matplotlib import pyplot as plt
import numpy as np


def decode(predictions):
    """
    Decodes predictions with labels and confidence values.

    :param predictions: A 2D numpy array of length 3, with output from Model-11 output neurons.
    :return: Returns a translated version of predictions, with labels.
    """
    decoded_predictions = []
    result_possibilities = ["Healthy", "Recurrent GBM", "Early Stage GBM"]
    for index in range(len(predictions[0])):
        decoded_predictions.append([result_possibilities[index], predictions.tolist()[0][index]])

    return decoded_predictions


def preprocess(image):
    """
    Processes an image so that it can fit Model-11 input neurons.

    :param image: A 2D numpy array.
    :return: An image preprocessed to fit Model-11 input neurons.
    """
    processed_image = scipy.misc.imresize(image, (256, 256))
    return processed_image.reshape(1, 256, 256, 1)


def view_image(image):
    """
    Displays a given image.

    :param image: A 2D numpy array
    """

    plt.imshow(image.astype(np.float32), cmap='gray')
    plt.show()
