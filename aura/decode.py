import scipy.misc
from matplotlib import pyplot as plt
import numpy as np

def decode(predictions):
    decoded_predictions = []
    result_possibilities = ["Healthy", "Early GBM", "Early GBM"]
    for index in range(len(predictions[0])):
        decoded_predictions.append([result_possibilities[index], predictions.tolist()[0][index]])

    return decoded_predictions

def preprocess(image):
    processed_image = scipy.misc.imresize(image, (256, 256))
    return processed_image.reshape(1,256,256,1)

def view_image(image):
    plt.imshow(image.astype(np.float32), cmap='gray')
    plt.show()
