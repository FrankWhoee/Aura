import nibabel as nib
import numpy
import os
from matplotlib import pyplot

T1 = nib.load("../Aura_Data/Healthy/102816_3T_Structural_1.6mm_preproc/T1w_restore.1.60.nii")
ArrayNIFTI = T1.get_data()
print(T1.shape)
# Display images using pyplot.
for i in range(113):
    pyplot.figure(dpi=300)
    pyplot.axes().set_aspect('equal', 'datalim')
    pyplot.set_cmap(pyplot.gray())
    img = numpy.flipud(ArrayNIFTI[:, :, i]);
    # pyplot.hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
    print("Displaying", i, "out of", 113)
    pyplot.pcolormesh(numpy.flipud(ArrayNIFTI[:, :, i]), clim=(0.0,0.001))
    pyplot.show()
