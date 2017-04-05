import layers as lyrs
import initializers as init
import costs as cst
import network as ntw
import optimisers as opt
import dataset as dts

from tensorflow.examples.tutorials.mnist import input_data

# DEFINE THE NETWORK STRUCTURES
n_inputs = 784
n_outputs = 10

relu_units1 = 128
relu_units2 = 128


layers = list()
layers.append(lyrs.LinearLayer(relu_units1, initializer=init.NormalWeightsConstBias()))
layers.append(lyrs.ReluLayer())
layers.append(lyrs.LinearLayer(relu_units2, initializer=init.NormalWeightsConstBias()))
layers.append(lyrs.ReluLayer())
layers.append(lyrs.LinearLayer(n_outputs, initializer=init.ZeroInitializer))
layers.append(lyrs.SoftmaxLayer())

cost = cst.CrossEntropy()

network = ntw.Network(layers, cost, n_inputs, n_outputs)
optim = opt.GD(learning_rate=0.0001)


# CREATE DATASETS

# use tensorflow to download the mnist dataset
mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

# creates a dateset class, in this case a wrapper of tensorflow one
train_data = dts.MnistTfDataset(mnist.train)
test_data = dts.MnistTfDataset(mnist.test)

train_scores, test_scores = network.sgd_train(iterations=1000,
                                              optimiser=optim,
                                              batch_size=128,
                                              train_data=train_data,
                                              test_data=test_data,
                                              measure_training=True)

# print train_scores and test_scores if you want

network.save_model("test.ckpt")     # will save network object using pickle







