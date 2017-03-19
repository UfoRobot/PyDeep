from py_deep import layers as lyrs
from py_deep import initializers as init
from py_deep import costs as cst
from py_deep import network as ntw
from py_deep import optimisers as opt
import pickle


def model_a():
    output_size = 10
    linear_layer = lyrs.LinearLayer(output_size, initializer=init.ZeroInitializer)
    softmax_layer = lyrs.SoftmaxLayer()

    cost = cst.CrossEntropy()
    optim = opt.GD()

    return [linear_layer, softmax_layer], cost, optim


def model_b():
    relu_units = 128
    output_size = 10

    linear_layer1 = lyrs.LinearLayer(relu_units, initializer=init.NormalWeightsConstBias())
    relu_layer = lyrs.ReluLayer()
    linear_layer2 = lyrs.LinearLayer(output_size, initializer=init.ZeroInitializer)
    softmax_layer = lyrs.SoftmaxLayer()

    layers = [linear_layer1, relu_layer, linear_layer2, softmax_layer]
    cost = cst.CrossEntropy()
    optim = opt.GD()

    return layers, cost, optim


def model_c():
    relu_units1 = 128
    relu_units2 = 128
    output_size = 10

    linear_layer1 = lyrs.LinearLayer(relu_units1, initializer=init.NormalWeightsConstBias())
    relu_layer1 = lyrs.ReluLayer()
    linear_layer2 = lyrs.LinearLayer(relu_units2, initializer=init.NormalWeightsConstBias())
    relu_layer2 = lyrs.ReluLayer()
    linear_layer3 = lyrs.LinearLayer(output_size, initializer=init.ZeroInitializer)
    softmax_layer = lyrs.SoftmaxLayer()

    layers = [linear_layer1, relu_layer1, linear_layer2, relu_layer2, linear_layer3, softmax_layer]
    cost = cst.CrossEntropy()
    optim = opt.GD()

    return layers, cost, optim


def model_d():
    layers = list()
    layers.append(lyrs.Volumer(28, 28, 1))
    layers.append(lyrs.Conv2D([3, 3, 1, 16], init.ConvXavier([3, 3, 1, 16])))
    layers.append(lyrs.ReluLayer())
    layers.append(lyrs.MaxPool([2,2]))
    layers.append(lyrs.Conv2D([3, 3, 16, 16], init.ConvXavier([3, 3, 16, 16])))
    layers.append(lyrs.ReluLayer())
    layers.append(lyrs.MaxPool([2, 2]))
    layers.append(lyrs.Flattener(7, 7, 16))
    #layers.append(lyrs.LinearLayer(128, init.ReluXavier()))
    #layers.append(lyrs.ReluLayer())
    layers.append(lyrs.LinearLayer(10, init.LinearXavier()))
    layers.append(lyrs.SoftmaxLayer())
    cost = cst.CrossEntropy()
    optim = opt.GD()

    return layers, cost, optim


def model_builder(model):

    switcher = {
        'A': model_a,
        'B': model_b,
        'C': model_c,
        'D': model_d
    }

    layers, cost, optim = switcher[model]()
    return layers, cost, optim


class CustomModel:

    def __init__(self, model, restore=False):

        self.model = model

        if restore:
            self.restore()
        else:
            layers, cost, optim = model_builder(model)

            direct_backprop = True          # Implemented for all model
            self.network = ntw.Network(layers, cost, 784, 10, direct_backprop, name=model)
            self.optimiser = optim

    def predict(self, to_predict):
        return self.network.predict(to_predict)

    def train(self, training_iterations, learning_rate, batch_size, data_train, data_evaluate, plot_training=False, everyN=100):
        self.optimiser.learning_rate = learning_rate
        self.network.init_layers()
        self.network.sgd_train(training_iterations, self.optimiser, batch_size, data_train, data_evaluate,
                               plot=plot_training, everyN=everyN)

    def evaluate(self, evaluate_data):
        return self.network.compute_accuracy(evaluate_data.images, evaluate_data.labels)

    def save_model(self):
        self.network.clear_tmp_vars()
        save_path = 'saved/model_' + self.model + ".obj"
        file = open(save_path, "wb")
        pickle.dump(self.network, file)
        file.close()

    def restore(self):
        print('Restoring model...')
        save_path = 'saved/model_' + self.model + ".obj"
        file = open(save_path, 'rb')
        self.network = pickle.load(file)
        file.close()

    def close_model(self):
        pass
