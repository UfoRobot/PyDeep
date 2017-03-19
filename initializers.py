import numpy as np
from scipy.stats import truncnorm


def truncated_normal(mean, std, shape):
    """
    Implements a truncated normal sampler

    """
    truncated_sampler = lambda mu, sd : truncnorm.rvs((0 - mu) / sd, np.inf, mu, sd)
    truncated_sampler_vec = np.vectorize(truncated_sampler, otypes=[np.float])

    mean_matrix = np.multiply(mean, np.ones(shape))

    return truncated_sampler_vec(mean_matrix, std)


class ZeroInitializer:

    @staticmethod
    def init_weights(shape):
        return np.zeros(shape)

    @staticmethod
    def init_bias(shape):
        return np.zeros(shape)


class NormalInitializer:

    @staticmethod
    def init_weights(shape):
        return np.random.randn(shape)

    @staticmethod
    def init_bias(shape):
        return np.random.randn(shape[0], shape[1])


class NormalWeightsConstBias:

    def __init__(self, weight_mean=0, weight_std=0.1, bias_constant=0.1):
        """
        Initialize weights as truncated normal, bias as constant

        """
        self.weight_mean = weight_mean
        self.weight_std = weight_std
        self.bias_constant = bias_constant

    def init_weights(self, shape):
        return truncated_normal(self.weight_mean, self.weight_std, shape)

    def init_bias(self, shape):
        return np.multiply(self.bias_constant, np.ones(shape))


class ConvXavier:

    def __init__(self, shape):
        self.std = np.sqrt(shape[0] * shape[1] * shape[2])

    def init_weights(self, shape):
        return self.std * np.random.normal(size=shape)

    def init_bias(self, shape):
        return np.multiply(0.1, np.ones(shape))


class LinearXavier:

    def init_weights(self, shape):
        fan_out = shape[0]
        fan_in = shape[1]
        std = np.sqrt(2 / (fan_in + fan_out))
        return std * np.random.normal(size=shape)

    def init_bias(self, shape):
        return np.random.normal(size=shape)



class ReluXavier:

    def init_weights(self, shape):
        fan_in = shape[1]
        std = np.sqrt(2 / fan_in)
        return std * np.random.normal(size=shape)

    def init_bias(self, shape):
        return np.multiply(0.1, np.ones(shape))








