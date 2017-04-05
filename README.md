# PyDeep

A pure-python / numpy deep learning library

## Features
- Linear Transformation Layer
- Convolutional layer
- MaxPool layer
- RELU activation fucntion
- Sigmoid activation function 
- Cross Entropy cost
- Mini-Batch Gradient Descent

The code is flexible and easily extensible to implement new layers, costs or optimiser.

### Prerequisites

The Library has the following dependencies:

```
numpy
scipy
```

### Installing

Clone the repository in your project

```
git clone https://github.com/UfoRobot/PyDeep
```

in your project import PyDeep

```
from PyDeep import *
```

### Example
We train a network on the mnist dataset. The network layers will be:
- Linear trannsformation with 128 units and RELU activation
- Linear trannsformation with 128 units and RELU activation
- Linear transoformation with 10 units and Sigmoid activation

and we will use the Cross Entropy cost metrics with the Mini-Batch Stochastich Gradient Descent Optimiser.

```
from py_deep import layers as lyrs
from py_deep import initializers as init
from py_deep import costs as cst
from py_deep import network as ntw
from py_deep import optimisers as opt
import matplotlib.pyplot as plt                                   # To make the trainig plot 
from tensorflow.examples.tutorials.mnist import input_data        # To easily get the dataset if you have tensorflow already installed
                                                                  # If you don't, see at the line where we load the data

# DEFINE THE NETWORK STRUCTURES
n_inputs = 784      #mnist images are 28*28
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

# Define the final network
network = ntw.Network(layers, cost, n_inputs, n_outputs)

optim = opt.GD(learning_rate=0.0001)


# Now get the data, the network uses a dataset object, we provide a wrapper for the tensorflow object but you can easily make your own one by extending the Dataset abstract class and putting your data into it

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
train_data = dts.MnistTfDataset(mnist.train)
test_data = dts.MnistTfDataset(mnist.test)

# Now train, we can request measures of how the training is proceeding to further make a plot:
train_scores, test_scores = network.sgd_train(iterations=10000,
                                              optimiser=optim,
                                              batch_size=128,
                                              train_data=train_data,
                                              test_data=test_data,
                                              measure_training=True)

# Plot the measures from the training process
#.....
#

# Save model
network.save_model("test.ckpt")     # will save network object using pickle
```
