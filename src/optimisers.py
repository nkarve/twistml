import numpy as np
from layers import layer, ABC, abstractmethod

eps = 1e-8
class optimiser(ABC):
    '''
    Base class of optimisers for trainable parameters
    '''

    @abstractmethod
    def update(self, id, param, grad):
        """
        Update a single trainable parameter
        :param ids: Unique parameter identifier, for preserving information
        :param params: Trainable parameter tensor
        :param grad: Update gradients tensor
        """

        pass

    def __call__(self, ids, params, grads):
        """
        Update a set of trainable parameters
        :param ids: Tuple of unique parameter identifiers, for preserving information
        :param params: Tuple of trainable parameter tensors, in order of ids
        :param grads: Tuple of update gradient tensors with respect to the corresponding parameters, in order of ids
        """
        
        # map(lambda t: self.update(*t), zip(ids, params, grads))
        for id_, param, grad in zip(ids, params, grads):
            self.update(id_, param, grad)
        

class sgd(optimiser):
    def __init__(self, eta):
        """
        :param eta: Learning rate
        """
        self.eta = eta
    
    def update(self, id_, param, grad):
        param -= grad * self.eta

class momentum(optimiser):
    def __init__(self, eta, gamma):
        """
        :param eta: Learning rate
        :param gamma: Controls acceleration of gradient in a direction
        """
        
        self.eta = eta
        self.gamma = gamma
        self.v = {}

    def update(self, id_, param, grad):
        if id_ not in self.v:
            self.v[id_] = 0.
        self.v[id_] = self.gamma * self.v[id_] + self.eta * grad
        param -= self.v[id_]


class adagrad(optimiser):
    def __init__(self, eta):
        """
        :param eta: Learning rate
        """
        
        self.eta = eta
        self.d = {}

    def update(self, id_, param, grad):
        if id_ not in self.d:
            self.d[id_] = 0
        self.d[id_] += grad * grad
        param -= self.eta / np.sqrt(eps + self.d[id_]) * grad


class rmsprop(optimiser):
    def __init__(self, eta, gamma):
        """
        :param eta: Learning rate
        :param gamma: Controls acceleration of gradient in a direction
        """

        self.eta = eta
        self.gamma = gamma
        self.d = {}

    def update(self, id_, param, grad):
        if id_ not in self.d:
            self.d[id_] = 0.
        self.d[id_] = (1 - self.gamma) * grad * grad + self.gamma * self.d[id_]
        param -= self.eta / np.sqrt(eps + self.d[id_]) * grad


class adam(optimiser):
    def __init__(self, eta, gamma1=0.9, gamma2=0.999):
        """
        :param eta: Learning rate
        :param gamma1: Exponential decay for the first momentum estimate
        :param gamma2: Exponential decay for the second momentum estimate

        """

        self.eta = eta
        self.b = gamma1, gamma2
        self.d = {}

    def update(self, id_, param, grad):
        # Array indices for [first order momentum estimate, second order momentum estimate, epochs]
        m_, v_, t_ = 0, 1, 2

        if id_ not in self.d:
            self.d[id_] = [0., 0., 0.]
        self.d[id_][m_] = (1 - self.b[m_]) * grad + self.b[m_] * self.d[id_][m_]
        self.d[id_][v_] = (1 - self.b[v_]) * grad * grad + self.b[v_] * self.d[id_][v_]
        self.d[id_][t_] += 1

        mp = self.d[id_][m_] / (1 - self.b[m_] ** self.d[id_][t_])
        vp = self.d[id_][v_] / (1 - self.b[v_] ** self.d[id_][t_])

        param -= self.eta * mp / (eps + np.sqrt(vp))
