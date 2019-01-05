import time
import numpy
from matplotlib import pyplot
import os
import sys

path = input("Path of file to be read: ")
if path == "":
    print("No path entered. Defaulting to ../Aura_Data/{136x136x217000}Healthy.aura")
    path = "../Aura_Data/{136x136x221182}Healthy.aura"

while not os.path.isfile(path):
    sys.stderr.write("File does not exist.\n")
    time.sleep(0.01)
    print("Current working directory is: " + os.getcwd())
    path = input("Path of file to be read: ")

filename = path.split("/")
filename = filename[len(filename) - 1]
l,w,n = filename[filename.find("{") + 1 : filename.rfind("}")].split("x")
l,w,n = int(l),int(w), int(n)
print("Loading images...")
initial = time.time()
# Load unshaped array into numpy
unshapedArray = numpy.fromfile(path, dtype=numpy.float16)
# Determine number of images by dividing the length of the unshaped array by the area of each image.
num_of_images = int(len(unshapedArray)/(l*w))
if num_of_images != n:
    unshapedArray = numpy.fromfile(path)
    num_of_images = int(len(unshapedArray) / (l * w))
final = time.time()
difference = final - initial
print(num_of_images, "images loaded in", str(difference)[0:5], "seconds.")

# Reshape the array to a 3D matrix.
Array = unshapedArray.reshape(l, w, num_of_images)
print("Array shaped. Displaying", num_of_images , "images with dimensions", l, "x", w)

# Display images using pyplot.
for i in range(num_of_images):
    pyplot.figure(dpi=300)
    pyplot.axes().set_aspect('equal', 'datalim')
    pyplot.set_cmap(pyplot.gray())
    img = numpy.flipud(Array[:, :, i])
    print("Displaying", i, "out of", num_of_images)
    pyplot.pcolormesh(img)
    pyplot.show()
