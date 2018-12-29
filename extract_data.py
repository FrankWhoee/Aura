import pydicom as dicom
import os
import numpy
from matplotlib import pyplot
import sys

path_data = "../Aura_Data/RIDER NEURO MRI"
lstFilesDCM = []
i = 0
progress_bar = "[                    ] 0%"
for dirName, subdirList, fileList in os.walk(path_data) :
    for filename in fileList:
        if ".dcm" in filename.lower():
            sys.stdout.write('\r')
            bar = ""
            for x in range((int)((i/20060) * 20)):
                bar += "="
            for x in range((int)((1 - (i/20060)) * 20)):
                bar += " "
            progress_bar = "[" + bar + "]" + str((int)((i/20060) * 100)) + "%"
            sys.stdout.write(progress_bar)
            sys.stdout.flush()

            lstFilesDCM.append(os.path.join(dirName,filename))
            i += 1

sys.stdout.write('\r')
progress_bar = "[====================] 100%"
sys.stdout.write(progress_bar)
sys.stdout.flush()
print("\n" + str(lstFilesDCM.__sizeof__()) + " filenames read.")
lstFilesDCM.sort()

RefDs = dicom.read_file(lstFilesDCM[0])

ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

ArrayDicom = numpy.zeros((256,256,len(lstFilesDCM)), dtype=RefDs.pixel_array.dtype)

# loop through all the DICOM files
badFiles = 0
goodFiles = 0
i = 0
for filenameDCM in lstFilesDCM:
    # print("Extracting " + filenameDCM)
    # read the file
    ds = dicom.read_file(filenameDCM)

    # store the raw image data
    try:
        ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array
        goodFiles += 1
    except:
        badFiles += 1
        # sys.stderr.write("Exception occurred for file " + filenameDCM + "\n")
    sys.stdout.write('\r')
    bar = ""
    for x in range((int)((i / len(lstFilesDCM)) * 50)):
        bar += "="
    for x in range((int)((1 - (i / len(lstFilesDCM))) * 50)):
        bar += " "
    progress_bar = "[" + bar + "]" + str((int)((i / len(lstFilesDCM)) * 100)) + "%"
    sys.stdout.write(progress_bar)
    sys.stdout.flush()
    i += 1

ArrayDicom.tofile("RIDER_Data")
print("DATA EXTRACTION COMPLETE.")
print(badFiles, " bad files found and not read.")
print(goodFiles, " good files found and read.")

