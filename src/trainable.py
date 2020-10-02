from layers import layer, ABC, abstractmethod
import numpy as np
from numpy.lib.stride_tricks import as_strided

eps = 1e-8

class trainable(layer, ABC):
    """
    The base class for layers with trainable parameters
    """

    def set_params(self, *params):
        """
        Generates a unique identifier for each trainable parameter
        :param params: Tuple of distinct parameter names
        """
        
        hash_ = str(hash(self))
        self.params = tuple(map(lambda p: hash_ + ':' + p, params))

    @abstractmethod
    def update(self, opt):
        """
        Updates all trainable parameters
        :param opt: The optimiser object
        """
        pass


class affine(trainable):
    """
    Layer with densely connected neurons. Alternatively called
    Dense/Fully Connected layer
    """
     
    def __init__(self, in_, out_, init=np.random.randn):
        """
        :param in_: The input dimension
        :param out_: The output dimension
        :param init: The function to initialise the weight matrix
        """
    
        self.w = init(in_, out_)
        self.b = np.random.randn(out_)
        self.set_params('w', 'b')

    def __call__(self, x):
        self.x = x
        return x.dot(self.w) + self.b

    def back(self, dE_dy):
        self.dE_dw = self.x.T.dot(dE_dy) / self.x.shape[0]  
        self.dE_db = np.sum(dE_dy, axis=0) / self.x.shape[0]  
        
        return dE_dy.dot(self.w.T)

    def update(self, opt):
        opt(self.params, [self.w, self.b], [self.dE_dw, self.dE_db])


# This is what you came for

def conv4d_forward(x, f, s):
    """
    Forward pass through a convolutional layer

    :param x: Image tensor of dimensions (Batch size, Height, Width, Input Channels)
    :param f: Convolutional filter of dimensions 
    (Filter height, Filter width, Input Channels, Output Channels)
    :param s: Tuple of strides (Stride along height, Stride along width)
    :return: The convolved image of dimensions 
    (Batch size, 1 + (H - Fh) / Sh, 1 + (W - Fw) / Sw, Output Channels)
    """
    
    B, H, W, C = x.shape
    Fh, Fw, C, D = f.shape
    Sh, Sw = s

    strided_shape = B, 1 + (H - Fh) // Sh, 1 + (W - Fw) // Sw, Fh, Fw, C
    
    ''' 
    This converts the image to a 6-dimensional tensor, where the two extra dimensions represent 
    strided, windowed 'snapshots' of each image (along the H and W dimensions) for every batch 
    and channel 
    
    E.g. if each image is 5x5 pixels and the filter is 3x3xCxD, let 
    x[0, :, :, 0] = [[ 1.  2.  3.  4.  5.]
                     [ 6.  7.  8.  9. 10.]
                     [11. 12. 13. 14. 15.]
                     [16. 17. 18. 19. 20.]
                     [21. 22. 23. 24. 25.]]

    After the next line, this is converted to: 
    [[[[ 1.  2.  3.]
       [ 6.  7.  8.]
       [11. 12. 13.]]
    
      [[ 2.  3.  4.]
       [ 7.  8.  9.]
       [12. 13. 14.]]
    
      [[ 3.  4.  5.]
       [ 8.  9. 10.]
       [13. 14. 15.]]]
    
     [[[ 6.  7.  8.]
       [11. 12. 13.]
       [16. 17. 18.]]
    
      [[ 7.  8.  9.]
       [12. 13. 14.]
       [17. 18. 19.]]
    
      [[ 8.  9. 10.]
       [13. 14. 15.]
       [18. 19. 20.]]]
    
     [[[11. 12. 13.]
       [16. 17. 18.] 
       [21. 22. 23.]]
    
      [[12. 13. 14.]
       [17. 18. 19.]
       [22. 23. 24.]]
    
      [[13. 14. 15.]
       [18. 19. 20.]
       [23. 24. 25.]]]]
    
    i.e. mimicking a 3x3 filter sliding over the image. This is done for each batch and channel 
    '''

    x = as_strided(x, strided_shape,
                   strides=(x.strides[0], Sh * x.strides[1], Sw * x.strides[2], x.strides[1], x.strides[2], x.strides[3]))
    
    '''
    Now, the filter elements in the Fh, Fw and C dimensions are multiplied component-wise with the 
    image 'snapshots'. This is done for all D filters separately, and then we obtain the convolved image.
    The entire code is barely 6 lines long, with 3 lines merely for tuple unpacking.
    '''

    return np.einsum('wxyijk,ijkd->wxyd', x, f)


def conv4d_filter_gradient(x, f, dE_dy, s):
    """
    Compute the gradient of the loss with respect to the convolutional filter, given the 
    gradient with respect to the convolutional output

    :param x: Original convolutional input tensor
    :param f: Convolutional filter tensor
    :param dE_dy: Gradient tensor of the loss with respect to the output
    :param s: Tuple of strides 
    :return: Gradient tensor of the loss with respect to the filter
    """
    
    
    B, H, W, C = x.shape
    Fh, Fw, C, D = f.shape
    Sh, Sw = s
    strided_shape = B, 1 + (H - Fh) // Sh, 1 + (W - Fw) // Sw, Fh, Fw, C
    x = as_strided(x, strided_shape,
               strides=(x.strides[0], Sh * x.strides[1], Sw * x.strides[2], x.strides[1], x.strides[2], x.strides[3]))
    
    '''
    It's a little-known fact that if:

    y = einsum('abc,cde->ace', x, f) The summation can be arbitrary, but lets use this as an example

    then: dy/df = einsum('abc,ace->cde', x, ones_like(y)), where all we have to do is swap the output 
    indices with those of the variable whose gradient we want to calculate, and replace y with a tensor
    the same size as y, but containing all ones.

    If you think about it, differentiation involving matrix multiplications is just a special case
    of this. Anyway, we're not trying to calculate dy/df here; we're trying to find dE/df, given dE/dy.
    So by the chain rule, instead of ones_like(y), we put in dE/dy, and we're done!
    '''
    
    return np.einsum('wxyijk,wxyd->ijkd', x, dE_dy)


def conv4d_input_gradient(x, f, dE_dy, s):
    """
    Compute the gradient of the loss with respect to the convolutional input, given the 
    gradient with respect to the convolutional output

    :param x: Original convolutional input tensor
    :param f: Convolutional filter tensor
    :param dE_dy: Gradient tensor of the loss with respect to the output
    :param s: Tuple of strides 
    :return: Gradient tensor of the loss with respect to the input
    """
   
    '''
    Here we use the same trick as we did to calculate the filter gradient. However in this case, we don't
    recover the input gradient immediately after the einsum; rather we obtain a strided version (since the 
    original input was strided before it went into the einsum during the forward pass convolution). So we
    need to sum over all the sub-windows to recover the original shape.
    '''
    
    dE_dx_strided = np.einsum('wxyd,ijkd->wxyijk', dE_dy, f)
    imax, jmax, di, dj = dE_dx_strided.shape[1:5]
    Sh, Sw = s

    dE_dx = np.zeros_like(x)
    for i in range(0, imax):
        for j in range(0, jmax):
            dE_dx[:, Sh*i:Sh*i+di, Sw*j:Sw*j+dj, :] += dE_dx_strided[:, i, j, ...]


    return dE_dx


class conv4d(trainable):
    """
    Convolutional layer operating on 4-dimensional tensors
    """
    
    def __init__(self, f_shape, strides=(1, 1)):
        """
        :param f_shape: Tuple of (Filter height, Filter width, Number of input channels, Number of output channels)
        :param strides: Number of cells by which the filter shifts, 
        in the image height and width dimensions, defaults to (1, 1)
        """
        
        self.filter = np.random.randn(*f_shape)
        self.strides = strides
        self.set_params('f')

    def __call__(self, x):
        self.x = x
        return conv4d_forward(x, self.filter, self.strides)

    def back(self, dE_dy):
        self.dE_df = conv4d_filter_gradient(self.x, self.filter, dE_dy, self.strides) / self.x.shape[0]  
        return conv4d_input_gradient(self.x, self.filter, dE_dy, self.strides)

    def update(self, opt):
        opt(self.params, [self.filter], [self.dE_df])


class batch_norm(trainable):
    def __init__(self, channels):
        self.gamma = np.random.randn(channels)
        self.beta = np.random.randn(channels)
        self.set_params('gamma', 'beta')
    
    def __call__(self, x):
        self.x = x

        self.mu, self.std = np.mean(x, axis=0), np.sqrt(np.var(x, axis=0) + eps)
        xnorm = (x - self.mu) / self.std
        print(self.mu.shape, self.std.shape)
        return self.gamma * xnorm + self.beta
    
    def back(self, dE_dy):
        dE_dxnorm = self.gamma * dE_dy
        
        dE_dvar = np.sum(dE_dxnorm * (self.x - self.mu) * -0.5 * np.power(self.std, 3), axis=0)
        dE_dmu = -np.sum(dE_dxnorm / self.std, axis=0) + dE_dvar * -2 * np.mean(self.x - self.mu, axis=0)

        self.dE_dg = np.sum(dE_dy * dE_dxnorm)
        self.dE_db = np.sum(dE_dy)
        
        dE_dx = dE_dxnorm / self.std + dE_dvar * 2 * (self.x - self.mu) / dE_dy.shape[0] + dE_dmu / dE_dy.shape[0] 
        return dE_dx

    def update(self, opt):
        opt(self.params, [self.gamma, self.beta], [self.dE_dg, self.dE_db])
