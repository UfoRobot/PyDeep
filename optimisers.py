import abc


class Optimiser:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def optimise(self, param, grad):
        """
        Every layer optimiser MUST implement this method, which updates the parameters given the respective gradients

        """
        return


class GD(Optimiser):

    def __init__(self, learning_rate=0.5):
        """
        Implements plain gradient descent
        """
        self.learning_rate = learning_rate

    def optimise(self, param, grad):

        updated = []
        for p, g in zip(param, grad):
            p += self.learning_rate * g
            updated.append(p)

        return updated

    def rescale_learning_rate(self, factor):
        self.learning_rate = self.learning_rate / factor
