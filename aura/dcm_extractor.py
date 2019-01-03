import pydicom as dicom
import os, numpy, sys, time

sys.stderr.write("WARNING: All .dcm files must have the same image dimensions.\n")
time.sleep(0.01)
path_data = input("Path to folder containing all .dcm files: ")
newFilename = input("Filename to dump information into: ")
if ".aura" not in newFilename:
    newFilename += ".aura"
print("Saving to " + newFilename)
time.sleep(5)

lstFilesDCM = []
print("Reading path...")
for dirName, subdirList, fileList in os.walk(path_data) :
    for filename in fileList:
        if ".dcm" in filename.lower():
            print("Loaded " + filename)
            lstFilesDCM.append(os.path.join(dirName,filename))

print("\n" + str(lstFilesDCM.__sizeof__()) + " file names read.")
lstFilesDCM.sort()

RefDs = dicom.read_file(lstFilesDCM[0])

ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

x = numpy.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
y = numpy.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
z = numpy.arange(0.0, (ConstPixelDims[2]+1)*ConstPixelSpacing[2], ConstPixelSpacing[2])

ArrayDicom = numpy.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

# loop through all the DICOM files
print("Loading images into numpy array...")
badFiles = 0
goodFiles = 0
image_num = 0
progress_bar_length = 50;

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
        # sys.stderr.write("Exception occurred for file " + filenameDCM + " dimensions invalid; should match dimensions of first image.\n")
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

ArrayDicom.tofile(newFilename)
print("\n\n----------------------- DATA EXTRACTION COMPLETE. -----------------------")
print(badFiles, " bad files found and not read because of invalid dimensions.")
print(goodFiles, " good files found and read, with proper dimensions.")
print("Your matrix dimensions are (length, width, number of images): ", ConstPixelDims)

