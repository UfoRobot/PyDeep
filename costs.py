import abc
import numpy as np
from scipy.misc import logsumexp


class Cost:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute_cost(self, prediction, true):
        """
        Every cost MUST implement this method, which computes the cost.
        """

    @abc.abstractmethod
    def start_back_prop(self, prediction, true):
        """
        Every layer MUST implement this method, which computes the first dl_dy of the backprop chain

        """
        return


class CrossEntropy(Cost):

    def compute_cost(self, predictions, tests):
        # get index of true prediction
        trues = np.argmax(tests, axis=1)

        # get predicted value for it
        predictions_of_interest = predictions[np.arange(len(trues)), trues]

        # compute cross entropy
        return np.add(-predictions_of_interest, logsumexp(predictions_of_interest, axis=1))
    
    def start_back_prop(self, prediction, test):
        prediction += 0.000000000001     # for numerical stability in later division
        return np.divide(test, prediction).transpose()

    def start_back_prop_direct(self, last_layer, prediction, test):
        if type(last_layer).__name__ == "SoftmaxLayer":
            return test - prediction
        else:
            raise ValueError("Direct back propagation not supported for this combo")
