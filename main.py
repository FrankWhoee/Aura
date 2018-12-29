import pydicom as dicom
import os
import numpy
from matplotlib import pyplot, cm

path_dicom = "./Data/QIN LUNG CT/QIN-LSC-0003/04-01-2015-1-CT Thorax wContrast-41946/2-THORAX W  3.0  B41 Soft Tissue-71225"
lstFilesDCM = []
for dirName, subdirList, fileList in os.walk(path_dicom) :
    for filename in fileList:
        if ".dcm" in filename.lower():
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
for filenameDCM in lstFilesDCM:
    # read the file
    ds = dicom.read_file(filenameDCM)
    # store the raw image data
    ArrayDicom[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array

for i in range(67):
    print(lstFilesDCM[i])
    pyplot.figure(dpi=300)
    pyplot.axes().set_aspect('equal', 'datalim')
    # pyplot.set_cmap(pyplot.gray())
    print(x)
    print(y)
    print(ArrayDicom[:, :, i])
    pyplot.pcolormesh(x, y, numpy.flipud(ArrayDicom[:, :, i]))
    pyplot.show()

