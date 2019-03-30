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
for dirName, subdirList, fileList in os.walk(path_data):
    for filename in fileList:
        if ".nii" in filename.lower():
            print("Loaded " + filename)
            lstFilesNii.append(os.path.join(dirName, filename))

print("\n" + str(len(lstFilesNii)) + " file names read.")
lstFilesNii.sort()

img = nib.load(lstFilesNii[2])
img_data = img.get_fdata()


@DeprecationWarning
def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


x, y, z = img_data.shape[0:3]
array_size = max(x, y, z)
dimensions = (array_size, array_size, ((x + y + z) * len(lstFilesNii)))
array_nifti = np.zeros(dimensions, dtype=numpy.float16)

newFilename = "{" + str(dimensions[0]) + "x" + str(dimensions[1]) + "x" + str(dimensions[2]) + "}" + newFilename
print("Saving to " + newFilename)
sys.stderr.write("[WARN] Keep numbers within {}. Those are important dimensions key to loading the file later.\n")

# loop through all the NIfTI files
print("Loading images into numpy array...")
badFiles = 0
goodFiles = 0
image_num = 0
progress_bar_length = 50;
last_index = 0;

initial_time = time.time()

for filenameNii in lstFilesNii:
    # read the file
    img = nib.load(filenameNii)
    img_data = img.get_fdata()

    if len(img_data.shape) > 3:
        img_data = img_data[:, :, :, 0]
    x, y, z = img_data.shape[0:3]

    for i in range(x):
        array_nifti[:, :, i + last_index] = eu.convert_to_size(img_data[i, :, :], (array_size, array_size))

    for i in range(y):
        array_nifti[:, :, i + x + last_index] = eu.convert_to_size(img_data[:, i, :], (array_size, array_size))

    for i in range(z):
        array_nifti[:, :, i + x + y + last_index] = eu.convert_to_size(img_data[:, :, i], (array_size, array_size))

    final_time = time.time()
    duration = final_time - initial_time
    last_index += x + y + z
    sys.stdout.write('\r')
    bar = ""
    for x in range((int)((image_num / len(lstFilesNii)) * progress_bar_length)):
        bar += "▮"
    for x in range((int)((1 - (image_num / len(lstFilesNii))) * progress_bar_length)):
        bar += " "
    progress_bar = "[" + bar + "]" + str(((image_num / len(lstFilesNii)) * 100))[0:5] + "% (" + str(duration)[
                                                                                                0:5] + "s)"
    sys.stdout.write(progress_bar)
    sys.stdout.flush()
    image_num += 1

final_time = time.time()
duration = final_time - initial_time
array_nifti.tofile(newFilename)
print("\n\n----------------------- DATA EXTRACTION COMPLETE. -----------------------")
print("Your matrix dimensions are (length, width, number of images): ", dimensions)
print("Extraction completed in " + str(duration)[0:5] + "s")