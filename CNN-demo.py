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
import scipy.misc

root = "../Aura_Data/";
cancerPath = root + "ChunkedCancerTestset/"
healthyPath = root + "ChunkedHealthyTestset/"
cancerSize = "{256x256x270}"
healthySize = "{136x136x181}"

cl,cw,cn = pAD(cancerSize)
hl,hw,hn = pAD(healthySize)
fl, fw = max(cl, cw, hl, hw), max(cl, cw, hl, hw)
fn = cn + hn
num_classes = 2

model = Sequential()

# Convolutional layers and Max pooling
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(256,256,1)))
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
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(1024, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.load_weights("Model-v6.hf")

cancerous_test_data = read_file(path=cancerPath + cancerSize + "Chunk5.aura").T
healthy_test_data = read_file(path=healthyPath + healthySize + "Chunk5.aura")
healthy_test_data = reshape(healthy_test_data, (fl,fw, hn)).T
test_data = np.zeros((fn, fl,fw))
for i in range(cn):
    test_data[i] = cancerous_test_data[i]
for i in range(hn):
    test_data[i + cn] = healthy_test_data[i]

x_test = test_data

x_test = test_data.reshape(fn,fl,fw,1)

image = test_data[50]
# image = eu.convertToSize(image,(136,136))
# image = scipy.misc.imresize(image, (256, 256))

plt.imshow(image)
plt.show()

print(model.predict(image.reshape(1,256,256,1)))