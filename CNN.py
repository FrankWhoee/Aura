from __future__ import print_function
import time
import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import random
import scipy.misc
import matplotlib.pyplot as plt
from aura.extractor_util import reshape
from aura.extractor_util import parseAuraDimensions as pAD
from aura.aura_loader import read_file
from keras.models import load_model
from time import time

print("Modules imported.")
print(os.getcwd())
root = "../Aura_Data/";
cancerPath = root
healthyPath = root
tumorPath = root

# cancerSize = "{256x256x3511}"
# healthySize = "{136x136x2353}"
# tumorSize = "{256x256x5501}"

cancerTrainSize = "{256x256x63198}"
healthyTrainSize = "{136x136x199063}"
cancerTestSize = "{256x256x7021}"
healthyTestSize = "{136x136x22118}"

tumorSize = "{256x256x250}"

trainSuffix = "Trainset"
testSuffix = "Testset"
cancerPrefix = "RIDER"
healthyPrefix = "Healthy"
tumorPrefix = "Tumor"
fileExtension = ".aura"


# This function takes in a list of paths to extract data and converts it to a numpy array.
def get_data(training_data_paths):
    """
    :param training_data_paths: a list of paths from which to extract data, shapes must be (l,w,n)
    :return: two numpy arrays with shuffled data, shape of (n,l,w), of data type numpy.float16 and a numpy array of shape (n) with labels

    n: number of images

    l: length of each image

    w: width of each image
    """
    init_time = time()
    print("Retrieving data from " + str(training_data_paths.__len__()) + " paths.")
    sizes = []
    l, w = pAD(training_data_paths[0][training_data_paths[0].find("{"):training_data_paths[0].find("}") + 1])[0:2]
    for filename in training_data_paths:
        print("Recording dimensions of " + filename)
        """
        fl: file length
        fw: file width
        fn: file number of images
        """
        fl, fw, fn = pAD(filename[filename.find("{"):filename.find("}") + 1])
        if fl > l:
            l = fl
        if fw > w:
            w = fw
        sizes.append(fn)
    n = sum(sizes)
    print(str(n) + " images found.")
    # train_data is a numpy array of (n,l,w) with data type numpy.float16
    train_data = np.zeros((n, l, w), dtype=np.float16)

    # Load in all data
    print("Loading data.")
    data = []
    for size,path in enumerate(training_data_paths):
        raw_data = read_file(path=path)
        raw_data = reshape(raw_data, (l, w, sizes[size])).T
        data.append(raw_data)

    # Compile data[] into output
    print("Compiling data into one array.")
    index_of_train_data = 0
    for index, package in enumerate(data):
        for image in package:
            train_data[index_of_train_data] = image
            index_of_train_data += 1

    # Label training data
    print("Labelling data.")
    data = []
    index_of_train_data = 0
    for size_index in range(sizes.__len__()):
        for index in range(sizes[size_index]):
            data.append((train_data[index_of_train_data], size_index))
            index_of_train_data += 1

    print("Shuffling data.")
    random.shuffle(data)

    print("Separating labels.")
    # Separate training images and labels
    labels = np.zeros(n)
    train_data = np.zeros((n, l, w))
    for i, (data, label) in enumerate(data):
        train_data[i] = data
        labels[i] = label

    final_time = time()
    duration = final_time - init_time
    print("Data retrieval complete. Process took " + str(duration) + " seconds.")
    return train_data, labels


# # Prepare paths
# root = "../Aura_Data/"
# train_paths = [root + "{136x136x199063}HealthyTrainset.aura", root + "{256x256x63198}RIDERTrainset.aura", root + "{256x256x7918}BTPTrainset.aura"]
# test_paths = [root + "{136x136x22118}HealthyTestset.aura", root + "{256x256x7021}RIDERTestset.aura",  root + "{256x256x879}BTPTestset.aura"]

# # Prepare paths
root = "../Aura_Data/Chunked/Dataset/"
train_paths = [root + "{136x136x181}HealthyTrainset.aura", root + "{256x256x270}CancerTrainset.aura"]
test_paths = [root + "{136x136x181}HealthyTestset.aura", root + "{256x256x270}CancerTestset.aura"]

train_data, train_label = get_data(train_paths)
test_data, test_label = get_data(test_paths)

train_n, train_l, train_w = train_data.shape
test_n, test_l, test_w = test_data.shape

# Set up CNN
batch_size = 32
num_classes = 3
epochs = 10
# input image dimensions
img_rows, img_cols = train_l, train_w

y_train = train_label.copy()
y_test = test_label.copy()

x_train = train_data.reshape(train_n, train_l, train_w, 1)
x_test = test_data.reshape(test_n, test_l, test_w, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(str(num_classes) + " classes set.")
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

# Convolutional layers and Max pooling
model.add(Conv2D(32, kernel_size=(16, 16),
                 activation='relu',
                 input_shape=(train_l, train_w, 1)))
model.add(MaxPooling2D(pool_size=(8, 8)))
model.add(Conv2D(64, (4, 4), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(128, (2, 2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (2, 2), activation='relu'))

# Dense layers and output
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# checkpoint
filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit(x_train, y_train,
                    callbacks=callbacks_list,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
finish_time = str(time.time())
model.save("model" + finish_time[:finish_time.find(".")] + ".hf")
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(history.history['loss'])
print(history.history['loss'])
print(history.history['val_loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
