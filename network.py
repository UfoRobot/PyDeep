import numpy as np
import pickle


class Network:

    def __init__(self, layers, cost, n_inputs, n_outputs):
        self.cost = cost
        self.layers = layers
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.init_layers()

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

        :param input_data [batch x dim]
        """

        layer_input = input_data
        for layer in self.layers:
            layer_input = layer.forward_pass(layer_input)

        return layer_input

    def sgd_train(self, iterations, optimiser, batch_size, train_data, test_data=None, measure_training=False, everyN=100):
        # Rescale learning rate according to batch size
        """
        Performs gradient descent using the optimiser and the parameters of choice.

        :param iterations: number of training iterations
        :param optimiser: optimiser
        :param batch_size: batch size
        :param train_data: train data, as a Dataset object
        :param test_data: test data to measure performance during training
        :param measure_training: boolean, if measure performance on test set during training
        :param everyN: frequency of training performance measurement

        :return: 2 lists with the performance during training on the train and test set. Empty lists if no measurement.
        """

        print("Training using gradient descent")
        optimiser.rescale_learning_rate(batch_size)

        test_scores = []
        train_scores = []

        for iteration in range(iterations):
            train_x, train_y = train_data.next_batch(batch_size)

            prediction = self.predict(train_x)
            self.back_prop(prediction, train_y, optimiser)

            if measure_training and iteration % everyN == 0:
                acc_train = self.compute_accuracy(train_data)
                acc_test = self.compute_accuracy(test_data)
                train_scores.append(acc_train)
                test_scores.append(acc_test)
                print("Iteration {},\t train accuracy: {},\t test accuracy: {}".format(iteration, acc_train, acc_test))

        return train_scores, test_scores

    def back_prop(self, prediction, test, optimiser):

        # Try direct backprop if implemented
        try:
            dl_dy = self.cost.start_back_prop_direct(self.layers[-1], prediction, test)
            layers_for_back_prop = self.layers[::-1][1:]
        except ValueError:
            dl_dy = self.cost.start_back_prop(prediction, test)
            layers_for_back_prop = self.layers[::-1]

        for layer in layers_for_back_prop:
            dl_dy = layer.backward_pass(dl_dy, optimiser)

    def compute_accuracy(self, dataset):
        images = dataset.get_images()
        labels = dataset.get_labels()

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

    def save_model(self, save_path):
        print("Saving model...")
        self.clear_tmp_vars()
        file = open(save_path, "wb")
        pickle.dump(self, file)
        file.close()



