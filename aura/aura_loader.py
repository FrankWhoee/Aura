import time
import numpy
from aura.extractor_util import parse_aura_dimensions
import numpy as np
from aura.extractor_util import reshape
from aura.extractor_util import parse_aura_dimensions as pAD
import random


def read_file(path):
    """
    Reads an aura file, converting it to numpy array.

    :param path: Path to aura file.
    :return: A numpy array.
    """
    filename = path.split("/")
    filename = filename[len(filename) - 1]
    l, w, n = parse_aura_dimensions(filename)
    print("Loading " + filename + "...")
    initial = time.time()

    # Load unshaped array into numpy
    unshaped_array = numpy.fromfile(path, dtype=numpy.float16);

    # Determine number of images by dividing the length of the unshaped array by the area of each image.
    num_of_images = int(len(unshaped_array) / (l * w))
    if num_of_images != n:
        unshaped_array = numpy.fromfile(path);
        num_of_images = int(len(unshaped_array) / (l * w))
    final = time.time()
    difference = final - initial
    print(num_of_images, "images loaded in", str(difference)[0:5], "seconds.")

    # Reshape the array to a 3D matrix.
    return unshaped_array.reshape(l, w, num_of_images)


# This function takes in a list of paths to extract data and converts it to a numpy array.
def get_data(training_data_paths, shuffle=True):
    """
    :param training_data_paths: a list of paths from which to extract data, shapes must be (l,w,n)
    :return: two numpy arrays with shuffled data, shape of (n,l,w), of data type numpy.float16 and a numpy array of shape (n) with labels

    n: number of images

    l: length of each image

    w: width of each image
    """
    init_time = time()
    print("Retrieving data from " + str(training_data_paths.__len__()) + " paths.")
    sizes = []
    l, w = pAD(training_data_paths[0][training_data_paths[0].find("{"):training_data_paths[0].find("}") + 1])[0:2]
    for filename in training_data_paths:
        print("Recording dimensions of " + filename)
        """
        fl: file length
        fw: file width
        fn: file number of images
        """
        fl, fw, fn = pAD(filename[filename.find("{"):filename.find("}") + 1])
        if fl > l:
            l = fl
        if fw > w:
            w = fw
        sizes.append(fn)
    n = sum(sizes)
    print(str(n) + " images found.")
    # train_data is a numpy array of (n,l,w) with data type numpy.float16
    train_data = np.zeros((n, l, w), dtype=np.float16)

    # Load in all data
    print("Loading data.")
    data = []
    for size, path in enumerate(training_data_paths):
        raw_data = read_file(path=path)
        raw_data = reshape(raw_data, (l, w, sizes[size])).T
        data.append(raw_data)

    # Compile data[] into output
    print("Compiling data into one array.")
    index_of_train_data = 0
    for index, package in enumerate(data):
        for image in package:
            train_data[index_of_train_data] = image
            index_of_train_data += 1

    # Label training data
    print("Labelling data.")
    data = []
    index_of_train_data = 0
    for size_index in range(sizes.__len__()):
        for index in range(sizes[size_index]):
            data.append((train_data[index_of_train_data], size_index))
            index_of_train_data += 1

    if shuffle:
        print("Shuffling data.")
        random.shuffle(data)

    print("Separating labels.")
    # Separate training images and labels
    labels = np.zeros(n)
    train_data = np.zeros((n, l, w))
    for i, (data, label) in enumerate(data):
        train_data[i] = data
        labels[i] = label

    final_time = time()
    duration = final_time - init_time
    print("Data retrieval complete. Process took " + str(duration) + " seconds.")
    return train_data, labels
