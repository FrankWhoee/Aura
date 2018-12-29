from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
mx.random.seed(1)

ctx = mx.cpu()

batch_size = 64
num_inputs = 784
num_outputs = 10
<<<<<<< HEAD

def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)

=======
def transform(data, label):
    return nd.transpose(data.astype(np.float32), (2,0,1))/255, label.astype(np.float32)
>>>>>>> 928c69e4a18074c99003f141f3539aaf903933eb
train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)
test_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=False, transform=transform),
                                     batch_size, shuffle=False)

num_fc = 512
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    # The Flatten layer collapses all axis, except the first one, into one axis.
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(num_fc, activation="relu"))
    net.add(gluon.nn.Dense(num_outputs))
<<<<<<< HEAD
net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
print("Layers set.")
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
print("Softmax cross entropy")
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
print("Trainer set.")
=======

net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})

>>>>>>> 928c69e4a18074c99003f141f3539aaf903933eb

def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]

<<<<<<< HEAD
print("Accuracy evaluator defined.")
epochs = 1
smoothing_constant = .01
print("\n\n----------------------- BEGINNING TRAINING -----------------------")

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        print("Training " , i , " out of " , train_data.__sizeof__() , ". (" , ((i/train_data.__sizeof__()) * 100) , "%)")
=======
epochs = 1
smoothing_constant = .01

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
>>>>>>> 928c69e4a18074c99003f141f3539aaf903933eb
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])
<<<<<<< HEAD
=======

>>>>>>> 928c69e4a18074c99003f141f3539aaf903933eb
        ##########################
        #  Keep a moving average of the losses
        ##########################
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))