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

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(256,256,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.load_weights("Model-v3.hf")

root = "../Aura_Data/";
cancerPath = root + "ChunkedCancerTestset/"
healthyPath = root + "ChunkedHealthyTestset/"
cancerSize = "{256x256x270}"
healthySize = "{136x136x181}"

cl,cw,cn = pAD(cancerSize)
hl,hw,hn = pAD(healthySize)
fl, fw = max(cl, cw, hl, hw), max(cl, cw, hl, hw)
fn = cn + hn

cancerous_test_data = read_file(path=cancerPath + cancerSize + "Chunk1.aura").T
healthy_test_data = read_file(path=healthyPath + healthySize + "Chunk1.aura")
healthy_test_data = reshape(healthy_test_data, (fl,fw, hn)).T
test_data = np.zeros((fn, fl,fw))
for i in range(cn):
    test_data[i] = cancerous_test_data[i]
for i in range(hn):
    test_data[i + cn] = healthy_test_data[i]

labels = np.zeros(fn)
for i in range(cn):
    labels[i] = 1

x_test = test_data
y_test = labels

x_test = test_data.reshape(fn,fl,fw,1)

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])