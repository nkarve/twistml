from layers import layer
from trainable import trainable
from loss import loss
from utils import make_batches, shuffle
import numpy as np

class model():
    """
    A unidirectional, sequential neural network model
    """
    
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, input_):
        result = input_
        for layer in self.layers:
            result = layer(result)
        return result

    def backpropagate(self, result, label, loss_fn, optimiser):
        error = loss_fn(label, result)  # / result.shape[0] - may have to do that
        grad = loss_fn.back(result)
        
        for layer in reversed(self.layers):
            grad = layer.back(grad)
            if isinstance(layer, trainable):
                layer.update(optimiser)

    def train(self, x_train, y_train, loss_fn, optimiser, metric, *, batch_size, epochs):
        for epoch in range(epochs):
            batches, labels = make_batches(x_train, y_train, batch_size, shuffle_data=True)

            for batch, label in zip(batches, labels):
                result = self(batch)
                self.backpropagate(result, label, loss_fn, optimiser)
            
            train_metric = metric(y_train, self(x_train))
            print(f'Epoch {epoch + 1}: {train_metric}')
    
    def test(self, x_test, y_test, metric):
        test_metric = metric(y_test, self(x_test))
        print(f'Test: {test_metric}')
