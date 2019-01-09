import numpy as np
import scipy

def convertToSize(input, size):
    image = scipy.misc.imresize(input, size)
    return image


def reshape(input,size):
    output = np.zeros(size)
    for image in range(output.shape[2]):
        output[:,:,image] = convertToSize(input[:,:,image],size[0:2])
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