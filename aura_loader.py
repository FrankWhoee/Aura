import time
import numpy

class aura_loader:

    def read_file(self, path, l, w):
        print("Loading images...")
        initial = time.time()
        # Load unshaped array into numpy
        unshapedArray = numpy.fromfile(path);
        # Determine number of images by dividing the length of the unshaped array by the area of each image.
        num_of_images = int(len(unshapedArray) / (l * w))
        final = time.time()
        difference = final - initial
        print(num_of_images, "images loaded in", str(difference)[0:5], "seconds.")

        # Reshape the array to a 3D matrix.
        ArrayDicom = unshapedArray.reshape(l, w, num_of_images)
        print("Array shaped. Displaying", num_of_images, "images with dimensions", l, "x", w)
        return ArrayDicom