import os
import sys
import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy
import numpy as np

from aura import extractor_util as eu

time.sleep(0.01)
path_data = input("Path to folder containing all NIfTI files: ")
if path_data == "":
    print("Defaulting to ../Aura_Data/Healthy/NIFTI")
    path_data = "../Aura_Data/Healthy/NIFTI"
newFilename = input("Filename to dump information into: ")
if ".aura" not in newFilename:
    newFilename += ".aura"
time.sleep(5)

lstFilesNii = []
print("Reading path...")
for dirName, subdirList, fileList in os.walk(path_data) :
    for filename in fileList:
        if ".nii" in filename.lower():
            print("Loaded " + filename)
            lstFilesNii.append(os.path.join(dirName, filename))

print("\n" + str(lstFilesNii.__sizeof__()) + " file names read.")
lstFilesNii.sort()

img = nib.load(lstFilesNii[2])
img_data = img.get_fdata()
def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

x, y, z = img_data.shape[0:3]
array_size = max(x, y, z)
dimensions = (array_size,array_size,217000)
ArrayNifti = np.zeros(dimensions, dtype=numpy.float16)

newFilename = "{" + str(dimensions[0]) + "x" + str(dimensions[1]) + "x" + str(dimensions[2]) + "}" + newFilename
print("Saving to " + newFilename)
sys.stderr.write("[WARN] Keep numbers within {}. Those are important dimensions key to loading the file later.\n")

# loop through all the NIfTI files
print("Loading images into numpy array...")
badFiles = 0
goodFiles = 0
image_num = 0
progress_bar_length = 50;

for filenameNii in lstFilesNii:
    # print("Extracting " + filenameDCM)
    # read the file
    img = nib.load(filenameNii)
    img_data = img.get_fdata()

    if len(img_data.shape) > 3:
        img_data = img_data[:,:,:,0]
    x, y, z = img_data.shape[0:3]
    for i in range(x):
        ArrayNifti[:, :, i] = eu.convertToSize(img_data[i, :, :], (array_size, array_size))

    for i in range(y):
        ArrayNifti[:, :, i + x] = eu.convertToSize(img_data[:, i, :], (array_size, array_size))

    for i in range(z):
        ArrayNifti[:, :, i + x + y] = eu.convertToSize(img_data[:, :, i], (array_size, array_size))
    sys.stdout.write('\r')
    bar = ""
    for x in range((int)((image_num / len(lstFilesNii)) * progress_bar_length)):
        bar += "▮"
    for x in range((int)((1 - (image_num / len(lstFilesNii))) * progress_bar_length)):
        bar += " "
    progress_bar = "[" + bar + "]" + str(((image_num / len(lstFilesNii)) * 100))[0:5] + "%"
    sys.stdout.write(progress_bar)
    sys.stdout.flush()
    image_num += 1

ArrayNifti.tofile(newFilename)
print("\n\n----------------------- DATA EXTRACTION COMPLETE. -----------------------")
print(badFiles, " bad files found and not read because of invalid dimensions.")
print(goodFiles, " good files found and read, with proper dimensions.")
print("Your matrix dimensions are (length, width, number of images): ", dimensions)





