import numpy as np
from numpy.lib.stride_tricks import as_strided
from abc import ABC, abstractmethod

eps = 1e-8
class layer(ABC):
    """
    The base class for layers with trainable and non-trainable parameters
    """
    
    @abstractmethod
    def __call__(self, x):
        """
        Forward pass through the layer
        :param x: The layer input tensor
        :return: A tensor produced by the layer operation
        """

    @abstractmethod
    def back(self, dE_dy):
        """
        Backward pass through the layer by the chain rule. If layer(x) = y, then layer.back(dE/dy) = dE/dx.
        :param dE_dy: The gradient received from the following layer
        :return: The gradient with respect to the input during the most recent forward pass
        """
        pass

class sigmoid(layer):
    """
    The sigmoid or logistic function
    """
    
    def __call__(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def back(self, dE_dy):
        return dE_dy * self.y * (1 - self.y)

class tanh(layer):
    """
    The hyperbolic tangent function    
    """
   
    def __call__(self, x):
        self.y = np.tanh(x)
        return self.y

    def back(self, dE_dy):
        return dE_dy * (1 - np.square(self.y))

class relu(layer):
    """
    The rectified linear unit (ReLU) function
    """
    def __call__(self, x):
        self.x = x
        return np.maximum(0, x)

    def back(self, dE_dy):
        return dE_dy * np.heaviside(self.x, 0.5)


class leaky_relu(layer):
    """
    The leaky rectified linear unit function. Similar to ReLU, but allows
    a small, positive gradient when the unit is not active
    """
    
    def __init__(self, a): self.a = a
    
    def __call__(self, x):
        self.x = x
        return np.maximum(self.a * x, x)

    def back(self, dE_dy):
        return dE_dy * np.where(self.x > 0, 1, self.a)
        
class softmax(layer):
    """
    The softmax or normalised exponential function. Returns a probability
    distribution over a vector, weighted according to their exponentials
    """
    
    def __call__(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.y = e_x / np.sum(e_x, axis=-1, keepdims=True)
        return self.y

    def back(self, dE_dy):
        J = np.apply_along_axis(lambda y: np.diag(y) - np.outer(y, y), -1, self.y)
        return np.einsum('...j,...jk->...k', dE_dy, J)


class flatten(layer):
    """
    Combines the dimensions of a tensor to a single dimension, 
    except the batch dimension.
    """
    
    def __call__(self, x):
        """
        :param x: The tensor to be flattened
        :return: The flattened, two-dimensional tensor 
        """
        
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def back(self, dE_dy):
        return dE_dy.reshape(self.x_shape)


class reshape(layer):
    """
    Changes the dimensions of a tensor
    """
    
    def __init__(self, shape):
        """
        :param shape: The shape to which to convert input tensors
        """
        
        self.shape = shape

    def __call__(self, x):
        """
        :param x: The tensor to be reshaped
        :return: The reshaped tensor
        """

        self.x_shape = x.shape
        return x.reshape((x.shape[0], self.shape))

    def back(self, dE_dy):
        return dE_dy.reshape(self.x_shape)
