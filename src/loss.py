from abc import ABC, abstractmethod
from utils import inv_one_hot
import numpy as np

eps = 1e-8
class loss(ABC):
    """
    Base class for all loss functions
    """

    @abstractmethod
    def __call__(self, y, yhat):
        """
        Calculate loss/cost/error (not normalised by batch size)
        :param y: Ground truth vector
        :param yhat: Predicted output vector
        :return: Error vector
        """
        pass

class l2(loss):
    """
    Euclidean norm error
    """
    
    def __call__(self, y, yhat):
        self.y = y
        return .5 * (y - yhat).dot((y - yhat).T)

    def back(self, yhat):
    	return yhat - self.y


class binary_cross_entropy(loss):
    """
    Binary cross entropy (simplifies categorical cross entropy
    for two classes). Also used for binary classification (e.g.
    true/false)
    """
    
    def __call__(self, y, yhat):
	    self.y = y
	    return -np.sum(y * np.log(yhat) + (1 - y) * np.log(1 - yhat + eps), axis=1)

    def back(self, yhat):
	    return (yhat - self.y) / (yhat * (1 - yhat) + eps)


class cross_entropy(loss):
    """
    Categorical cross entropy loss
    """
    
    def __call__(self, y, yhat):
        self.y = y
        return -np.sum(y * np.log(yhat + eps))
    
    def back(self, yhat):
        return -self.y / yhat


class cross_entropy_onehot(loss):
    """
    Categorical cross entropy loss in cases where ground truth vectors
    are one hot vectors (all 0s except the true label which is a 1)
    """
    
    def __call__(self, y, yhat):
        self.y = y
        return -np.log(yhat[np.where(y)])

    def back(self, yhat):
	    return -self.y / yhat
    

class loss_metric():
    """
    Base class for all loss metrics, used to compute the accuracy of a neural network
    """

    @abstractmethod
    def __call__(self, labels, result):
        """
        Compute the accuracy of a neural network, based on a custom methodology. It does
        NOT compute the loss/error and is thus not used for backpropagation; only for 
        debugging, display and analysis.

        :param label: Tensor containing ground truth values for the neural network
        :param result: Output tensor as computed by the neural network.
        :return: A scalar accuracy based on a metric
        """
        pass

class binary_accuracy(loss_metric):
    """
    Computes percentage of scalar outputs from a sigmoid binary classifier that match the given labels 
    """

    def __call__(self, labels, result):
        batch_size = result.shape[0]
        pct = np.count_nonzero(np.floor(result + 0.5) == labels) * 100. / batch_size
        return f'{pct:.2f}%'

class accuracy(loss_metric):
    """
    Computes percentage of one-hot outputs from a softmax n-class classifier that match the given labels 
    """

    def __call__(self, labels, result):
        batch_size = result.shape[0]
        choices = np.argmax(result, axis=-1)
        pct = np.count_nonzero(choices == np.where(labels == 1)[1]) * 100. / batch_size
        return f'{pct:.2f}%'

class top_n_accuracy(loss_metric):
    """
    Computes percentage of labels that match any one of the top n classes of one-hot outputs 
    from a softmax n-class classifier
    """

    def __init__(self, n):
        self.n = n

    def __call__(self, labels, result):
        batch_size = result.shape[0]
        nchoices = np.argpartition(result, -self.n, axis=-1)[:, -self.n:]
        pct = np.count_nonzero(nchoices == np.where(labels == 1)[1].reshape(-1, 1)) * 100. / batch_size
        return f'{pct:.2f}%'
