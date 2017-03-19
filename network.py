import numpy as np
import matplotlib.pyplot as plt


class Network:

    def __init__(self, layers, cost, n_inputs, n_outputs, direct_backprop, name="no_name"):
        self.cost = cost
        self.layers = layers
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.direct_backprop = direct_backprop
        self.init_layers()
        self.name = name

    def init_layers(self):
        """
        Initialise the size of the layers according to each layer output size and their chaining

        """

        layer_input = self.n_inputs
        for layer in self.layers:
            layer_input = layer.init_layer(layer_input)

        # Check for consistency
        if not layer_input == self.n_outputs:
            raise ValueError("Your architecture is not consistent with the layers you defined")

    def predict(self, input_data):
        """
        Predicts the output feed forwarding results through the layers

        """

        layer_input = input_data
        for layer in self.layers:
            layer_input = layer.forward_pass(layer_input)

        return layer_input

    def sgd_train(self, iterations, optimiser, batch_size, train_data, test_data, plot=False, everyN=100):
        # Rescale learning rate according to batch size
        optimiser.rescale_learning_rate(batch_size)

        test_scores = []
        train_scores = []



        for iteration in range(iterations):
            #print(iteration)
            train_x, train_y = train_data.next_batch(batch_size)

            prediction = self.predict(train_x)
            self.back_prop(prediction, train_y, optimiser)

            if plot and iteration % everyN == 0:
                acc_train = self.compute_accuracy(train_data.images, train_data.labels)
                acc_test = self.compute_accuracy(test_data.images, test_data.labels)
                train_scores.append(acc_train)
                test_scores.append(acc_test)
                print("Iteration {},\t train accuracy: {},\t test accuracy: {}".format(iteration, acc_train, acc_test))


                # Plot training and test error during training for selected parameter values
                plt.plot([i for i in range(0, iteration+1, 10) if i % everyN == 0], [1 - i for i in train_scores], 'b-',
                         label="Training error")
                plt.plot([i for i in range(0, iteration+1, 10) if i % everyN == 0], [1 - i for i in test_scores], 'r-',
                         label="Test error")

                plt.xlabel('Iteration')
                plt.ylabel('Validation')
                plt.legend()
                name = "plots/plot" + self.name + "2valid"
                plt.savefig(name)
                plt.clf()

    def back_prop(self, prediction, test, optimiser):

        if self.direct_backprop:
            dl_dy = self.cost.start_back_prop_direct(self.layers[-1], prediction, test)
            layers_for_back_prop = self.layers[::-1][1:]
        else:
            dl_dy = self.cost.start_back_prop(prediction, test)
            layers_for_back_prop = self.layers[::-1]

        for layer in layers_for_back_prop:
            dl_dy = layer.backward_pass(dl_dy, optimiser)

    def compute_accuracy(self, images, labels):
        # get index of true prediction
        trues = np.argmax(labels, axis=1)

        # get predicted value for it
        predictions = self.predict(images)
        predicted = np.argmax(predictions, axis=1)

        # compute cross entropy
        return np.sum((trues == predicted) * 1) / np.size(images, axis=0)

    def clear_tmp_vars(self):
        for layer in self.layers:
            layer.clear_tmp_vars()



