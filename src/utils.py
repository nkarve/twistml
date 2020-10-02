import numpy as np

def sqrt_variance(in_, out_): return np.random.randn(in_, out_) / np.sqrt(in_) 
def he(in_, out_): return np.random.randn(in_, out_) * np.sqrt(2 / in_)
def glorot(in_, out_): 
    return np.random.normal(0, np.sqrt(2 / (in_ + out_)), (in_, out_))

def one_hot(x): return np.eye(np.max(x) + 1)[x]
def inv_one_hot(x, labels): return labels[np.argmax(x, axis=-1)].flatten()

def shuffle(x, y):
    """
    Shuffles two tensors the same way along their first axes

    :param x: First tensor
    :param y: Second tensor
    :return: Tuple containing shuffled tensors
    """
    
    indices = np.random.permutation(x.shape[0])
    return x[indices], y[indices]

def make_batches(x, y, batch_size, shuffle_data=True): 
    """
    Divides the feature vectors and corresponding labels into equally sized chunks for batch processing

    :param x: Feature vector
    :param y: Label vector
    :param batch_size: Number of feature/label vectors in each batch
    :param shuffle: Whether to shuffle x and y before batching, defaults to True
    :return: A 2-tuple of batched feature and label tensors
    """
    
    if shuffle_data: x, y = shuffle(x, y)
    return [x[i:i + batch_size] for i in range(0, x.shape[0], batch_size)], \
        [y[i:i + batch_size] for i in range(0, y.shape[0], batch_size)]


def partition_dataset(x_train, y_train, x_test, y_test, n, shuffle_data=True):
    """
    Shuffles and redistributes training and test sets according to a specified size 

    :param x_train: Training feature vectors
    :param y_train: Training labels
    :param x_test: Test feature vectors
    :param y_test: Test labels
    :param n: Size of training dataset (if int); Percentage of items in training dataset (if float)
    :param shuffle: Whether to shuffle x and y before redistributing, defaults to True
    :return: A 4-tuple containing the redistributed dataset
    """
    
    size = x_train.shape[0] + x_test.shape[0]
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    if shuffle_data:
        indices = np.random.permutation(size)
        x, y = x[indices], y[indices]

    if isinstance(n, float):
        n = int(n * size)

    return x[:n], y[:n], x[n:], y[n:]
