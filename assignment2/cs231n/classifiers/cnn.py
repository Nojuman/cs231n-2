import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.randn(num_filters * (H/2) * (W/2), hidden_dim) * weight_scale
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params['b3'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    l1, l1_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    l2, l2_cache = affine_relu_forward(l1, W2, b2)
    l3, l3_cache = affine_forward(l2, W3, b3)

    scores = l3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))

    dl2, dW3, db3 = affine_backward(dscores, l3_cache)
    dl1, dW2, db2 = affine_relu_backward(dl2, l2_cache)
    dX, dW1, db1 = conv_relu_pool_backward(dl1, l1_cache)

    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3

    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W3'] = dW3
    grads['b3'] = db3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads

# new defined class  
class MyConvNet(object):
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=5,
               hidden_dim=100, num_classes=10, weight_scale=1e-4, reg=1e-3,
               dtype=np.float32):

    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    C, H, W = input_dim
    self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.randn(num_filters, num_filters, filter_size, filter_size) * weight_scale
    self.params['b2'] = np.zeros(num_filters)
    self.params['W3'] = np.random.randn(num_filters * (H/4) * (W/4), hidden_dim) * weight_scale
    self.params['b3'] = np.zeros(hidden_dim)
    self.params['W4'] = np.random.randn(hidden_dim, hidden_dim) * weight_scale
    self.params['b4'] = np.zeros(hidden_dim)
    self.params['W5'] = np.random.randn(hidden_dim, num_classes) * weight_scale
    self.params['b5'] = np.zeros(num_classes)
    self.params['gamma1'] = np.ones((num_filters, ), dtype=np.float32)
    self.params['gamma2'] = np.ones((num_filters, ), dtype=np.float32)
    self.params['gamma3'] = np.ones((hidden_dim, ), dtype=np.float32)
    self.params['gamma4'] = np.ones((hidden_dim, ), dtype=np.float32)
    self.params['beta1'] = np.zeros((num_filters, ), dtype=np.float32)
    self.params['beta2'] = np.zeros((num_filters, ), dtype=np.float32)
    self.params['beta3'] = np.zeros((hidden_dim, ), dtype=np.float32)
    self.params['beta4'] = np.zeros((hidden_dim, ), dtype=np.float32)

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    mode = 'test' if y is None else 'train'

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    gamma1, beta1 = self.params['gamma1'], self.params['beta1']
    gamma2, beta2 = self.params['gamma2'], self.params['beta2']
    gamma3, beta3 = self.params['gamma3'], self.params['beta3']
    gamma4, beta4 = self.params['gamma4'], self.params['beta4']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    bn_params = [{'mode': mode} for x in xrange(4)]

    l1, l1_cache = conv_batchnorm_relu_pool_forward(X, W1, b1, gamma1, beta1, conv_param, pool_param, bn_params[0])
    l2, l2_cache = conv_batchnorm_relu_pool_forward(l1, W2, b2, gamma2, beta2, conv_param, pool_param, bn_params[1])
    l3, l3_cache = affine_batchnorm_relu_forward(l2, W3, b3, gamma3, beta3, bn_params[2])
    l4, l4_cache = affine_batchnorm_relu_forward(l3, W4, b4, gamma4, beta4, bn_params[3])
    scores, l5_cache = affine_forward(l4, W5, b5)
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) \
        + np.sum(W3**2) + np.sum(W4**2) + np.sum(W5**2))

    dl4, dW5, db5 = affine_backward(dscores, l5_cache)
    dl3, dW4, db4, dgamma4, dbeta4 = affine_batchnorm_relu_backward(dl4, l4_cache)
    dl2, dW3, db3, dgamma3, dbeta3 = affine_batchnorm_relu_backward(dl3, l3_cache)
    dl1, dW2, db2, dgamma2, dbeta2 = conv_batchnorm_relu_pool_backward(dl2, l2_cache)
    dX, dW1, db1, dgamma1, dbeta1 = conv_batchnorm_relu_pool_backward(dl1, l1_cache)

    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    dW4 += self.reg * W4
    dW5 += self.reg * W5

    grads['W1'] = dW1
    grads['b1'] = db1
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W3'] = dW3
    grads['b3'] = db3
    grads['W4'] = dW4
    grads['b4'] = db4
    grads['W5'] = dW5
    grads['b5'] = db5
    grads['gamma1'] = dgamma1
    grads['gamma2'] = dgamma2
    grads['gamma3'] = dgamma3
    grads['gamma4'] = dgamma4
    grads['beta1'] = dbeta1
    grads['beta2'] = dbeta2
    grads['beta3'] = dbeta3
    grads['beta4'] = dbeta4
    
    return loss, grads

############### helper layers ###############
def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_param):
  a, fc_cache = affine_forward(x, w, b)
  temp, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(temp)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache

def affine_batchnorm_relu_backward(dout, cache):
  fc_cache, bn_cache, relu_cache = cache
  dtemp = relu_backward(dout, relu_cache)
  da, dgamma, dbeta = batchnorm_backward_alt(dtemp, bn_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db, dgamma, dbeta

def conv_batchnorm_relu_pool_forward(x, w, b, gamma, beta, conv_param, pool_param, bn_param):
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  temp, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  s, relu_cache = relu_forward(temp)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, bn_cache, relu_cache, pool_cache)
  return out, cache

def conv_batchnorm_relu_pool_backward(dout, cache):
  conv_cache, bn_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  dtemp = relu_backward(ds, relu_cache)
  da, dgamma, dbeta = spatial_batchnorm_backward(dtemp, bn_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, dgamma, dbeta
  
pass
