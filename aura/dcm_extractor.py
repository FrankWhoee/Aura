import pydicom as dicom
import os, numpy, sys, time
import scipy.misc

# path_data = input("Path to folder containing all .dcm files: ")
# new_filename = input("Filename to dump information into: ")
# resize_l = input("Length to resize images to:")
# resize_w = input("Width to resize images to:")

path_data = "../../Aura_Data/Unextracted/CPTAC-GBM"
new_filename = "CPTAC"
resize_l = 256
resize_w = 256

if ".aura" not in new_filename:
    new_filename += ".aura"

lstFilesDCM = []
print("Reading path...")
for dirName, subdirList, fileList in os.walk(path_data) :
    for filename in fileList:
        if ".dcm" in filename.lower():
            print("Loaded " + filename)
            lstFilesDCM.append(os.path.join(dirName,filename))

print("\n" + str(len(lstFilesDCM)) + " file names read.")
lstFilesDCM.sort()

RefDs = dicom.dcmread(lstFilesDCM[0])

ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])
ConstPixelDims = (resize_l, resize_w, len(lstFilesDCM))
array_dicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

new_filename = "{" + str(array_dicom.shape[0]) + "x" + str(array_dicom.shape[1]) + "x" + str(array_dicom.shape[2]) + "}" + new_filename
print("Saving to " + new_filename)

# loop through all the DICOM files
print("Loading images into numpy array...")
image_num = 0
progress_bar_length = 50

for filenameDCM in lstFilesDCM:
    ds = dicom.dcmread(filenameDCM)
    try:
        array_dicom[:, :, lstFilesDCM.index(filenameDCM)] = scipy.misc.imresize(ds.pixel_array, (256, 256))
    except:
        print(ds.pixel_array.shape)
    sys.stdout.write('\r')
    bar = ""
    for x in range((int)((image_num / len(lstFilesDCM)) * progress_bar_length)):
        bar += "▮"
    for x in range((int)((1 - (image_num / len(lstFilesDCM))) * progress_bar_length)):
        bar += " "
    progress_bar = "[" + bar + "]" + str((int)((image_num / len(lstFilesDCM)) * 100)) + "%"
    sys.stdout.write(progress_bar)
    sys.stdout.flush()
    image_num += 1

array_dicom.tofile(new_filename)
print("\n\n----------------------- DATA EXTRACTION COMPLETE. -----------------------")
