import time
import numpy
from matplotlib import pyplot, cm

path = input("Path of file to be read: ")
if path == "":
    print("No path entered. Defaulting to ../Aura_Data/3D_FLAIR_RIDER.aura")
    print("Default dimensions are 256,216\n")
    path = "../Aura_Data/3D_FLAIR_RIDER.aura"

f = open(path);
# Defaults to 3D_FLAIR_RIDER dimensions if nothing is inputted.
try:
    l = int(input("Length of image: "))
    w = int(input("Width of image: "))
except:
    print("Defaulting to 256,216\n")
    l,w = 256,216

print("Loading images...")
initial = time.time()
# Load unshaped array into numpy
unshapedArray = numpy.fromfile(path);
# Determine number of images by dividing the length of the unshaped array by the area of each image.
num_of_images = int(len(unshapedArray)/(l*w))
final = time.time()
difference = final - initial
print(num_of_images, "images loaded in", str(difference)[0:5], "seconds.")

# Reshape the array to a 3D matrix.
ArrayDicom = unshapedArray.reshape(l,w,num_of_images)
print("Array shaped. Displaying", num_of_images , "images with dimensions", l, "x", w)

# Display images using pyplot.
for i in range(num_of_images):
    pyplot.figure(dpi=300)
    pyplot.axes().set_aspect('equal', 'datalim')
    pyplot.set_cmap(pyplot.gray())
    img = numpy.flipud(ArrayDicom[:, :, i]);
    # pyplot.hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    print("Displaying", i, "out of", num_of_images)
    pyplot.pcolormesh(numpy.flipud(ArrayDicom[:, :, i]), clim=(0.0,0.001))
    pyplot.show()

