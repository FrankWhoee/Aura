from __future__ import print_function
import time
import os
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy
from aura.extractor_util import reshape
from matplotlib import pyplot as plt
from aura.extractor_util import parseAuraDimensions as pAD
from aura.aura_loader import read_file
from aura import extractor_util as eu
import pydicom as dcm
import PIL
from PIL import Image
import scipy.misc


root = "../Aura_Data/";

num_classes = 2

model = Sequential()

fl,fw = 256,256

# Convolutional layers and Max pooling
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(fl,fw,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(1024, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


# Dense layers and output
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.698736))
model.add(Dense(2048, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.load_weights("Model-v7.hf")
# image = dcm.read_file(root + "ChunkedHealthyTestset/{136x136x181}Chunk0.aura").pixel_array
# image = dcm.read_file(root + "CPTAC-GBM/C3L-00016/11-15-1999-MR BRAIN WOW CONTRAST-47088/8-AX 3D SPGR-43615/000199.dcm").pixel_array
image = dcm.read_file(root + "CPTAC-GBM/C3L-00016/11-15-1999-MR BRAIN WOW CONTRAST-47088/8-AX 3D SPGR-43615/000199.dcm").pixel_array


image = scipy.misc.imresize(image, (256, 256))

plt.imshow(image)
plt.show()

print(model.predict(image.reshape(1,256,256,1)))