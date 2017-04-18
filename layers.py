import abc
import numpy as np
from scipy.misc import logsumexp


class Layer:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def init_layer(self, n_inputs):
        """
        Every layer MUST implement this method, which initializes the layer given its input size,
        which corresponds to the previous layer output size.

        """
        return

    @abc.abstractmethod
    def forward_pass(self, layer_input):
        """
        Every layer MUST implement this method, which process the layer input to produce the output, and accepts multiple
        inputs as a batch

        """
        return

    @abc.abstractmethod
    def backward_pass(self, dl_dy, optimiser):
        """
        Every layer MUST implement this method, which updates the layer parameters using SGD
        and returns dl_dx to be fed as dl_dy to previous layer

        :return: dl_dx
        """
        return

    def clear_tmp_vars(self):
        pass



class LinearLayer(Layer):
    def __init__(self, n_units, initializer):
        """
        Create a linear layer with n_units output nodes

        """
        self.n_units = n_units
        self.initializer = initializer

        self.last_batch_input = np.zeros([1, 1])

        # bias and weights to be initialized once the architecture is completed
        self.weights = np.zeros(1)
        self.bias = np.zeros(1)

    def init_layer(self, n_inputs):
        self.weights = self.initializer.init_weights([self.n_units, n_inputs])
        self.bias = self.initializer.init_bias([self.n_units, 1])
        return self.n_units

    def forward_pass(self, layer_input):
        self.last_batch_input = layer_input
        return np.dot(layer_input, self.weights.transpose()) + self.bias.transpose()

    def backward_pass(self, dl_dy, optimiser):
        dl_dw = np.dot(dl_dy.T, self.last_batch_input)
        dl_db = dl_dy.sum(axis=0, keepdims=True).T
        dl_dx = np.dot(dl_dy, self.weights)

        # update params
        self.weights, self.bias = optimiser.optimise([self.weights, self.bias], [dl_dw, dl_db])
        return dl_dx

    def clear_tmp_vars(self):
        self.last_batch_input = None



class SoftmaxLayer(Layer):

    def __init__(self):
        """
        Softmax layer
        """
        self.n_units = 0
        self.last_output = np.zeros([1, 1])
        self.last_batch_output = np.zeros([1, 1])

    def init_layer(self, n_inputs):
        self.n_units = n_inputs
        return n_inputs

    def forward_pass(self, layer_input):
        log_normalized = np.subtract(layer_input, logsumexp(layer_input, axis=1, keepdims=True))
        self.last_batch_output = np.exp(log_normalized)
        return self.last_batch_output

    def backward_pass(self, dl_dy, optimiser):
        self.last_output += 0.000000000001    # add little noise for numerical stability

        temp = -np.dot(self.last_output, self.last_output.transpose())
        temp = np.add(temp, np.multiply(self.last_output, np.eye(self.n_units)))
        temp = np.dot(dl_dy, temp)
        return temp

    def clear_tmp_vars(self):
        self.last_batch_output = None


class ReluLayer(Layer):

    def __init__(self):
        """
        RELU layer
        """
        self.last_batch_output = np.zeros([1, 1])

    def init_layer(self, n_inputs):
        return n_inputs

    def forward_pass(self, layer_input):
        layer_input[layer_input < 0] = 0
        self.last_batch_output = layer_input
        return self.last_batch_output

    def backward_pass(self, dl_dy, optimiser):
        tmp = (self.last_batch_output > 0) * 1
        return np.multiply(tmp, dl_dy)

    def clear_tmp_vars(self):
        self.last_batch_output = None


class Flattener(Layer):

    def __init__(self, heigth, width, channels):
        """
        Flattens an image batch-tensor to a design matrix. The image batch-tensor follows the dimensionality convention:
        batch x channels x height x width

        Args:
            heigth:
            width:
            channels:
        """
        self.b = 0
        self.h = heigth
        self.w = width
        self.c = channels

    def init_layer(self, n_inputs):
        return self.c * self.h * self.w

    def forward_pass(self, layer_input):
        # save shape to convert the backward message into image batch-tensor convention
        self.b, self.c, self.h, self.w = layer_input.shape
        return layer_input.reshape(self.b, self.h * self.w * self.c)

    def backward_pass(self, dl_dy, optimiser):
        return dl_dy.reshape(self.b, self.c, self.h, self.w)


class Volumer(Layer):

    def __init__(self, heigth, width, channels):
        self.b = 0
        self.h = heigth
        self.w = width
        self.c = channels

    def init_layer(self, n_inputs):
        return None

    def forward_pass(self, layer_input):
        return layer_input.reshape(-1, self.c, self.h, self.w)

    def backward_pass(self, dl_dy, optimiser):
        return dl_dy.reshape(-1, self.c * self.h * self.w)



class Conv2D(Layer):

    def __init__(self, shape, initialiser, stride=1):
        """
        DOC ME
        Args:
            shape:
            stride:
            initialiser:
        """
        self.h_filter = shape[0]
        self.w_filter = shape[1]
        self.c_in = shape[2]
        self.c_out = shape[3]

        assert shape[0] == shape[1], "Filter must be square"

        self.initialiser = initialiser

        # Compute padding: (padding = 'SAME')
        self.padding = (self.h_filter - 1) // 2
        self.stride = stride

        self.last_input_splash = None
        self.out_shape = None

        self.weights = None
        self.bias = None


    def init_layer(self, n_inputs):
        self.weights = self.initialiser.init_weights([self.c_out, self.c_in, self.h_filter, self.w_filter])
        self.bias = self.initialiser.init_bias([self.c_out, 1])
        return None

    def forward_pass(self, layer_input):
        self.out_shape = layer_input.shape
        b, _, h, w = layer_input.shape

        # Convert input and weights to splash form
        X_splash = self.__splash(layer_input, self.h_filter, self.w_filter, padding=self.padding, stride=self.stride)
        self.last_input_splash = X_splash
        W_splash = self.weights.reshape(self.c_out, -1)
        out = W_splash @ X_splash + self.bias

        # Convert output into batch image tensor convention
        out = out.reshape(self.c_out, h, w, b)
        out = out.transpose(3, 0, 1, 2)

        return out

    def backward_pass(self, dl_dy, optimiser):
        dL_db = np.sum(dl_dy, axis=(0, 2, 3))
        dL_db = dL_db.reshape(self.c_out, -1)

        dL_dy_splash = dl_dy.transpose(1, 2, 3, 0).reshape(self.c_out, -1)
        dL_dw = dL_dy_splash @ self.last_input_splash.T
        dL_dw = dL_dw.reshape(self.weights.shape)

        W_splash = self.weights.reshape(self.c_out, -1)
        dL_dx_splash = W_splash.T @ dL_dy_splash

        dL_dx = self.__unsplash(dL_dx_splash, self.out_shape, self.h_filter, self.w_filter, padding=self.padding,
                                stride=self.stride)

        #update params
        self.weights, self.bias = optimiser.optimise([self.weights, self.bias], [dL_dw, dL_db])

        return dL_dx

    def __im2col_indices(self, x_shape, field_height, field_width, padding=1, stride=1):
        B, C, H, W = x_shape

        out_height = int((H + 2 * padding - field_height) / stride + 1)
        out_width = int((W + 2 * padding - field_width) / stride + 1)

        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

        return (k, i, j)

    def __splash(self, x, field_height, field_width, padding=1, stride=1):
        """
        Splashes a batch image-tensor into a 2d matrix using im2col channel wise and batchwise

        """
        # Zero-pad the input
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        k, i, j = self.__im2col_indices(x.shape, field_height, field_width, padding, stride)

        cols = x_padded[:, k, i, j]
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
        return cols

    def __unsplash(self, cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
        """
        Unsplashes a 2d matrix into a batch image-tensor

        """
        N, C, H, W = x_shape
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
        k, i, j = self.__im2col_indices(x_shape, field_height, field_width, padding,
                                        stride)
        cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        if padding == 0:
            return x_padded
        return x_padded[:, :, padding:-padding, padding:-padding]

    def clear_tmp_vars(self):
        self.last_input_splash = None


class MaxPool(Layer):

    def __init__(self, patch_shape, stride=None):
        """
        Implements a max pool layer. Note that the maxpool requires that the patch shape is square, and that the operation
        tiles the input.
        Args:
            patch_shape: [heigth, width], must be square
            stride: 
        """
        
        assert patch_shape[0] == patch_shape[1], "Patch shape must be square"

        self.pool_height = patch_shape[0]
        self.pool_width = patch_shape[1]
        
        if stride is not None:
            self.stride = stride
        else:
            self.stride = patch_shape[0]
            
        self.last_input_shape = None
        self.last_input_reshaped = None
        self.last_output = None

    def forward_pass(self, layer_input):
        """
        A fast implementation of the forward pass for the max pooling layer that uses
        some clever reshaping.

        This can only be used for square pooling regions that tile the input.

        """
        b, c, h, w = layer_input.shape
        self.last_input_shape = layer_input.shape

        self.last_input_reshaped = layer_input.reshape(b, c, h / self.pool_height, self.pool_height,
                                                       w / self.pool_width, self.pool_width)

        self.last_output = self.last_input_reshaped.max(axis=3).max(axis=4)

        return self.last_output


    def backward_pass(self, dl_dy, optimiser):
        """
        If there are multiple argmaxes, this method will assign gradient to
        ALL argmax elements of the input rather than picking one.
        """
        x_splashed, out = self.last_input_reshaped, self.last_output

        dl_dx = np.zeros_like(x_splashed)
        out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
        mask = (x_splashed == out_newaxis)
        dl_dy_new_axis = dl_dy[:, :, :, np.newaxis, :, np.newaxis]
        dl_dy_broadcast, _ = np.broadcast_arrays(dl_dy_new_axis, dl_dx)
        dl_dx[mask] = dl_dy_broadcast[mask]
        dl_dx /= np.sum(mask, axis=(3, 5), keepdims=True)
        dl_dx = dl_dx.reshape(self.last_input_shape)

        return dl_dx

    def clear_tmp_vars(self):
        self.last_input_reshaped = None
        self.last_output = None

