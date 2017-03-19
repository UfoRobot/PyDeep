# PyDeep

A pure-python / numpy deep learning library

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

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

n_inputs = 784    #mnist images are 28*28
n_outputs = 10

relu_units1 = 128
relu_units2 = 128


linear_layer1 = lyrs.LinearLayer(relu_units1, initializer=init.NormalWeightsConstBias())
relu_layer1 = lyrs.ReluLayer()
linear_layer2 = lyrs.LinearLayer(relu_units2, initializer=init.NormalWeightsConstBias())
relu_layer2 = lyrs.ReluLayer()
linear_layer3 = lyrs.LinearLayer(n_outputs, initializer=init.ZeroInitializer)
softmax_layer = lyrs.SoftmaxLayer()

# The layers structure is defined as a list of consecutive layers
layers = [linear_layer1, relu_layer1, linear_layer2, relu_layer2, linear_layer3, softmax_layer]

cost = cst.CrossEntropy()

# Define the final network
network = ntw.Network(layers, cost, n_inputs, n_outputs)

optim = opt.GD()

# Wrap the network into the model, which exposes API for saving and restoring as well
model = model.Model(network, optim, name="Example")

# Now get the data, if you don't have tensorflow you can easily load the train and test input images and labels your way,
# if you have tensorflow installed instead you can easily do:
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_input = mnist.train.images        # N_train x 729 numpy array
train_output = mnist.train.labels       # N_train x 10  numpy array
test_input = mnist.test.images          # N_test  x 729 numpy array
test_output = mnist.test.labels         # N_test  x 10  numpy array

data_train = [train_input, train_output]
data_test = [test_input, test_output]

# Now train, we can request measures of how the training is proceeding to further make a plot:
iterations = 5000
learning_rate = 0.001
batch_size = 250
measure_during_training_every_n = 100
train_measures, test_measures = network.train(iterations, learning_rate, batch_size, data_train, data_test, measure_during_training_every_n)

# Plot the measures from the training process
plt.plot([i for i in range(0, iterations+1, 10) if i % measure_during_training_every_n == 0], [1 - i for i in train_scores], 'b-',
         label="Training error")
plt.plot([i for i in range(0, iterations+1, 10) if i % measure_during_training_every_n == 0], [1 - i for i in test_scores], 'r-',
         label="Test error")

plt.xlabel('Iteration')
plt.ylabel('Error')
plt.legend()
plt.show()

```
