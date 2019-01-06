from __future__ import print_function
import time
import os
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


from aura.extractor_util import reshape
from aura.extractor_util import parseAuraDimensions as pAD
from aura.aura_loader import read_file

print("Modules imported.")
print(os.getcwd())
root = "../Aura_Data/";
cancerPath = root + "ChunkedRIDER/"
healthyPath = root + "ChunkedHealthy/"

cancerSize = "{256x256x3511}"
healthySize = "{136x136x2353}"

cl,cw,cn = pAD(cancerSize)
hl,hw,hn = pAD(healthySize)
fl, fw = max(cl, cw, hl, hw), max(cl, cw, hl, hw)
fn = cn + hn
# Set up data
cancerous_train_data = read_file(path=cancerPath + cancerSize + "Chunk0.aura").T
healthy_train_data = read_file(path=healthyPath+ healthySize + "Chunk0.aura")
healthy_train_data = reshape(healthy_train_data, (fl,fw,hn)).T
train_data = np.zeros((fn, fl,fw))
for i in range(cn):
    train_data[i] = cancerous_train_data[i]
for i in range(hn):
    train_data[i + cn] = healthy_train_data[i]
print(train_data.shape)

cancerous_test_data = read_file(path=cancerPath + cancerSize + "Chunk1.aura").T
healthy_test_data = read_file(path=healthyPath + healthySize + "Chunk1.aura")
healthy_test_data = reshape(healthy_test_data, (fl,fw, hn)).T
test_data = np.zeros((fn, fl,fw))
for i in range(cn):
    test_data[i] = cancerous_test_data[i]
for i in range(hn):
    test_data[i + cn] = healthy_test_data[i]
print(test_data.shape)

labels = np.zeros(fn)
for i in range(cn):
    labels[i] = 1

batch_size = 8
num_classes = 2
epochs = 2

# input image dimensions
img_rows, img_cols = fl,fw

y_train = labels.copy()
y_test = labels.copy()

x_train = train_data.reshape(fn,fl,fw,1)
x_test = test_data.reshape(fn,fl,fw,1)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(fl,fw,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
finish_time = str(time.time())
model.save("model"+finish_time[:finish_time.find(".")]+".hf")
print('Test loss:', score[0])
print('Test accuracy:', score[1])