import pydicom as dicom
import os
import numpy
from matplotlib import pyplot
import sys

path_data = "../Aura_Data/IvyGAP"
lstFilesDCM = []
for dirName, subdirList, fileList in os.walk(path_data) :
    for filename in fileList:
        if ".dcm" in filename.lower():
            print("Loaded " + filename)
            lstFilesDCM.append(os.path.join(dirName,filename))

lstFilesDCM.sort()

RefDs = dicom.read_file(lstFilesDCM[0])

ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

# loop through all the DICOM files
badFiles = 0
goodFiles = 0
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
        sys.stderr.write("Exception occurred for file " + filenameDCM + "\n")

f = open("Data/IVY_GAP_EXTRACTED_DATA.txt","w+")
f.write(ArrayDicom.tobytes());
f.close()
strbad = str(badFiles)
sys.stderr.write(strbad + " bad files found and not read.")
print(goodFiles, " good files found and read.")



for i in range(67):
    print(lstFilesDCM[i])
    pyplot.figure(dpi=300)
    pyplot.axes().set_aspect('equal', 'datalim')
    # pyplot.set_cmap(pyplot.gray())
    print(x)
    print(y)
    print(ArrayDicom[:, :, i])
    pyplot.pcolormesh(x, y, numpy.flipud(ArrayDicomLarge[:, :, i]))
    pyplot.show()

