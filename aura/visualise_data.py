import time
import numpy
from matplotlib import pyplot
import os
import sys
from aura.aura_loader import read_file as rf

path = input("Path of file to be read: ")
if path == "":
    print("No path entered. Defaulting to ../Aura_Data/{136x136x217000}Healthy.aura")
    path = "aura/{136x136x217000}Healthy.aura"

while not os.path.isfile(path):
    sys.stderr.write("File does not exist.\n")
    time.sleep(0.01)
    print("Current working directory is: " + os.getcwd())
    path = input("Path of file to be read: ")

Array = rf(path)

# Display images using pyplot.
for i in range(Array.shape[2]):
    pyplot.figure(dpi=300)
    pyplot.axes().set_aspect('equal', 'datalim')
    pyplot.set_cmap(pyplot.gray())
    img = numpy.flipud(Array[:, :, i])
    print("Displaying", i, "out of", Array.shape[2])
    pyplot.pcolormesh(img)
    pyplot.show()

