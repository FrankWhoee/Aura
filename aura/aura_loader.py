import time
import numpy
import os

def read_file(path):
    filename = path.split("/")
    filename = filename[len(filename) - 1]
    l, w, n = filename[filename.find("{") + 1: filename.rfind("}")].split("x")
    l, w, n = int(l), int(w), int(n)
    print("Loading images...")
    initial = time.time()
    # Load unshaped array into numpy
    unshapedArray = numpy.fromfile(path, dtype=numpy.float16);
    # Determine number of images by dividing the length of the unshaped array by the area of each image.
    num_of_images = int(len(unshapedArray) / (l * w))
    if num_of_images != n:
        unshapedArray = numpy.fromfile(path);
        num_of_images = int(len(unshapedArray) / (l * w))
    final = time.time()
    difference = final - initial
    print(num_of_images, "images loaded in", str(difference)[0:5], "seconds.")

    # Reshape the array to a 3D matrix.
    Array = unshapedArray.reshape(l, w, num_of_images)
    return Array