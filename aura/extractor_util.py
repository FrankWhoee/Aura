import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

def convertToSize(input, size):
    output = np.zeros(size)
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            if row < len(input) and col < len(input[row]):
                output[row][col] = input[row][col]

    return output


def reshape(input,size):
    output = np.zeros(size)
    for image in range(output.shape[2]):
        output[:,:,image] = scipy.misc.imresize(input[:,:,image],size[0:2])

    return output

def parseAuraDimensions(dimensions):
    l, w, n = dimensions[dimensions.find("{") + 1: dimensions.rfind("}")].split("x")
    l, w, n = int(l), int(w), int(n)
    return l,w,n

def stretch(image, minimum, maximum):
    image = (image - minimum) / (maximum - minimum)
    image[image < 0] = 0
    image[image > 1] = 1
    return image