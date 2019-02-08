from __future__ import print_function
from matplotlib import pyplot as plt
from aura.aura_loader import read_file
import scipy.misc
import numpy as np
from keras.models import load_model

root = "../Aura_Data/";

model = load_model("Model-10-1.hdf5")
# image = read_file(root + "ChunkedHealthyTestset/{136x136x181}Chunk1.aura").T[50]
imageCancer = read_file(root + "{256x256x7021}RIDERTestset.aura").T[5021]
imageHealthy = read_file(root + "{136x136x22118}HealthyTestset.aura").T[5021]
# image = dcm.read_file(root + "Unextracted/CPTAC-GBM/C3L-00016/11-15-1999-MR BRAIN WOW CONTRAST-47088/8-AX 3D SPGR-43615/000199.dcm").pixel_array


imageHealthy = scipy.misc.imresize(imageHealthy, (256, 256))
print(imageCancer)
imageCancer = scipy.misc.imresize(imageCancer, (256, 256))
print(imageCancer)
# plt.imshow(imageHealthy, cmap='gray')
# plt.show()
# print(type(imageCancer))
# plt.imshow(imageCancer.astype(np.float32), cmap='gray')
# plt.show()

print("Healthy prediction: " + str(model.predict(imageHealthy.reshape(1,256,256,1))))
print("Cancer prediction: " + str(model.predict(imageCancer.reshape(1,256,256,1))))
