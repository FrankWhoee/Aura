from __future__ import print_function
from matplotlib import pyplot as plt
from aura.aura_loader import read_file
import scipy.misc
from keras.models import load_model

root = "../Aura_Data/";

fl,fw = 256,256

num_classes = 3

model = load_model("Model-v8.hf")
# image = read_file(root + "ChunkedHealthyTestset/{136x136x181}Chunk1.aura").T[50]
image = read_file(root + "ChunkedCancerTestset/{256x256x270}Chunk1.aura").T[50]
# image = dcm.read_file(root + "Unextracted/CPTAC-GBM/C3L-00016/11-15-1999-MR BRAIN WOW CONTRAST-47088/8-AX 3D SPGR-43615/000199.dcm").pixel_array


image = scipy.misc.imresize(image, (256, 256))

plt.imshow(image)
plt.show()

print(model.predict(image.reshape(1,256,256,1)))

