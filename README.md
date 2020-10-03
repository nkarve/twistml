# twistml
## A fast, pedagogical, Pythonic machine learning library

### Features:
- Modular multi-layer neural networks with feedforward, convolutional, activation and utility layers
- Generic, fast, tensor-based backpropagation
- Support for several optimisers, from vanilla SGD to ADAM
- Extremely straightforward, compact, Pythonic style for learning and teaching
- 6 line CNN forward pass (3 of which are tuple unpackings!)

### Getting Started:
```python
import twist as tw
import numpy as np
import matplotlib.pyplot as plt

from twist.utils import partition_dataset, he, one_hot
from twist.trainable import affine, conv4d
from twist.network import model
from twist.layers import tanh, softmax, flatten
from twist.optimiser import optimiser

from emnist import extract_training_samples, extract_test_samples

np.random.seed(1)

(x_train, y_train), (x_test, y_test) = extract_training_samples('mnist'), extract_test_samples('mnist')
x_train, y_train, x_test, y_test = partition_dataset(x_train, y_train, x_test, y_test, .95, shuffle_data=True)

x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0
y_train, y_test = one_hot(y_train.astype(np.uint8)), one_hot(y_test.astype(np.uint8))

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

classifier = tw.network.model([
                conv4d((3, 3, 1, 3)),
                conv4d((3, 3, 3, 1)),
                flatten(),
                affine(576, 10),
                softmax()
             ])

loss = tw.loss.cross_entropy_onehot()
optimiser = tw.optimiser.adam(3e-3)
metric = tw.loss.accuracy()

classifier.train(x_train, y_train, loss, optimiser, metric, batch_size=64, epochs=32)
classifier.test(x_test, y_test, metric)
```
