import time
import numpy as np
import os
import sys
from aura.aura_loader import read_file as rf
from aura.extractor_util import convertToSize as cTS
import scipy.misc

path = input("Path of file to be read: ")
if path == "":
    print("No path entered. Defaulting to ../Aura_Data/Unchunked/{136x136x221182}Healthy.aura")
    path = "../../Aura_Data/Unchunked/{136x136x221182}Healthy.aura"

while not os.path.isfile(path):
    sys.stderr.write("File does not exist.\n")
    time.sleep(0.01)
    print("Current working directory is: " + os.getcwd())
    path = input("Path of file to be read: ")

array = rf(path)
newArray = np.zeros((256,256,1))
for image in range(array.shape[2]):
    newArray.append(scipy.misc.imresize(cTS(array[:,:,image], (136,136)), (256,256)))

newArray.tofile("{" + str(array.shape[0]) + "x" + str(array.shape[1]) + "x" + str(array.shape[2]) + "}Healthy.aura")

