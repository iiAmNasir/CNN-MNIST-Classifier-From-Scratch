#!/usr/bin/env python
# coding: utf-8

# ## Implementation of CNN from Scratch Using Numpy

# **Mount Google Drive and Set Current Directory to DLA4**

# In[54]:


from google.colab import drive
drive.mount('/content/drive')
get_ipython().run_line_magic('cd', '/content/drive/MyDrive/DLA4')


# **Import Useful Libraries**

# In[55]:



import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import sqrt, ceil
import cv2
from timeit import default_timer as timer


# # Forward Passes

# In[56]:


def cnn_forward_pass(x, w, b, cnn_params):
    stride = cnn_params['stride']
    pad = cnn_params['pad']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    cache = (x, w, b, cnn_params)
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    height_out = int(1 + (H + 2 * pad - HH) / stride)
    width_out = int(1 + (W + 2 * pad - WW) / stride)
    feature_maps = np.zeros((N, F, height_out, width_out))

    for n in range(N):
        for f in range(F):
            height_index = 0
            for i in range(0, H, stride):
                width_index = 0
                for j in range(0, W, stride):
                    feature_maps[n, f, height_index, width_index] =                         np.sum(x_padded[n, :, i:i+HH, j:j+WW] * w[f, :, :, :]) + b[f]
                    width_index += 1
                height_index += 1

    return feature_maps, cache


# # Testing Forward Pass for Convolutional Layer

# In[57]:



def absolute_error(x, y):
    return np.sum(np.abs(x - y))

x_shape = (1, 3, 4, 4)  
w_shape = (3, 3, 4, 4)  
b_shape = (3, )

x = np.linspace(0, 255, num=np.prod(x_shape), dtype='uint8').reshape(x_shape)
w = np.linspace(-1.0, 1.0, num=np.prod(w_shape), dtype='float64').reshape(w_shape)
b = np.linspace(-1.0, 1.0, num=np.prod(b_shape), dtype='float64').reshape(b_shape)

cnn_params = {'stride': 2, 'pad': 1}

out, _ = cnn_forward_pass(x, w, b, cnn_params)


print(out.shape)  
print()
print(out)
correct_out = np.array([[[[-1577.82517483, -1715.03496503],
   [-2154.29370629, -2308.0979021 ]],

  [[  480.12587413,   440.25874126],
   [  296.38461538,   240.59440559]],

  [[ 2538.07692308,  2595.55244755],
   [ 2747.06293706,  2789.28671329]]]])

print()
print(absolute_error(correct_out, out))  


# # Backward Pass for Convolutional Layer

# In[58]:



def cnn_backward_pass(derivative_out, cache):
    x, w, b, cnn_params = cache
    N, C, H, W = x.shape  
    F, _, HH, WW = w.shape  
    _, _, height_out, weight_out = derivative_out.shape  
    stride = cnn_params['stride']
    pad = cnn_params['pad']
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=0)

    for n in range(N):
        for f in range(F):
            for i in range(0, H, stride):
                for j in range(0, W, stride):
                    dx_padded[n, :, i:i+HH, j:j+WW] += w[f, :, :, :] * derivative_out[n, f, i, j]
                    dw[f, :, :, :] += x_padded[n, :, i:i+HH, j:j+WW] * derivative_out[n, f, i, j]
                    db[f] += derivative_out[n, f, i, j]

    dx = dx_padded[:, :, 1:-1, 1:-1]
    return dx, dw, db


# # Forward Pass for Max Pooling Layer

# In[59]:


def max_pooling_forward_pass(x, pooling_params):
    N, F, H, W = x.shape  

    pooling_height = pooling_params['pooling_height']
    pooling_width = pooling_params['pooling_width']
    stride = pooling_params['stride']


    cache = (x, pooling_params)


    height_pooled_out = int(1 + (H - pooling_height) / stride)
    width_polled_out = int(1 + (W - pooling_width) / stride)


    pooled_output = np.zeros((N, F, height_pooled_out, width_polled_out))


    for n in range(N):
        for i in range(height_pooled_out):
            for j in range(width_polled_out):
                ii = i * stride
                jj = j * stride
                current_pooling_region = x[n, :, ii:ii+pooling_height, jj:jj+pooling_width]
                pooled_output[n, :, i, j] =                     np.max(current_pooling_region.reshape((F, pooling_height * pooling_width)), axis=1)

    return pooled_output, cache


# # Testing Forward Pass for Max Pooling Layer

# In[60]:



def absolute_error(x, y):
    return np.sum(np.abs(x - y))



x_shape = (2, 1, 4, 4)  


x = np.linspace(0, 255, num=np.prod(x_shape), dtype='float64').reshape(x_shape)


pooling_params = {'pooling_height': 2, 'pooling_width': 2, 'stride': 2}


out, _ = max_pooling_forward_pass(x, pooling_params)


print(out.shape)  
print()
print(out)

correct_out = np.array([[[[ 41.12903226,  57.58064516],
   [106.93548387, 123.38709677]]],

 [[[172.74193548, 189.19354839],
   [238.5483871,  255.        ]]]])

print()
print(absolute_error(correct_out, out))  


# # Backward Pass for MAX Pooling Layer

# In[61]:




def max_pooling_backward_pass(derivatives_out, cache):
    x, pooling_params = cache
    N, F, H, W = x.shape

    pooling_height = pooling_params['pooling_height']
    pooling_width = pooling_params['pooling_width']
    stride = pooling_params['stride']
    height_pooled_out = int(1 + (H - pooling_height) / stride)
    width_polled_out = int(1 + (W - pooling_width) / stride)
    dx = np.zeros((N, F, H, W))

    for n in range(N):

        for f in range(F):
            for i in range(height_pooled_out):
                for j in range(width_polled_out):
                    ii = i * stride
                    jj = j * stride
                    current_pooling_region = x[n, f, ii:ii+pooling_height, jj:jj+pooling_width]
                    current_maximum = np.max(current_pooling_region)
                    temp = current_pooling_region == current_maximum
                    dx[n, f, ii:ii+pooling_height, jj:jj+pooling_width] +=                         derivatives_out[n, f, i, j] * temp
    return dx


# # Testing Backward Pass for Max Pooling Layer

# In[62]:


x_shape = (1, 1, 8, 8)  
derivatives_out_shape = (1, 1, 4, 4)  

x = np.linspace(0, 255, num=np.prod(x_shape), dtype='uint8').reshape(x_shape)
derivatives_out = np.random.randn(*derivatives_out_shape)
pooling_params = {'pooling_height': 2, 'pooling_width': 2, 'stride': 2}

out, cache = max_pooling_forward_pass(x, pooling_params)
dx = max_pooling_backward_pass(derivatives_out, cache)
print(x[0, 0, 0:2, 0:2])
print()
print(dx[0, 0, 0:2, 0:2])


# #Forward Pass for Fully-Connected Layer

# In[63]:



def fc_forward(x, w, b):
    cache = (x, w, b)
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1)
    fc_output = np.dot(x_reshaped, w) + b
    return fc_output, cache


# # Backward Pass for Fully-Connected Layer

# In[64]:



def fc_backward(derivatives_out, cache):
    x, w, b = cache
    dx = np.dot(derivatives_out, w.T).reshape(x.shape)
    N = x.shape[0]
    x = x.reshape(N, -1)
    dw = np.dot(x.T, derivatives_out)
    db = np.dot(np.ones(dx.shape[0]), derivatives_out)
    return dx, dw, db


# # Naive Forward Pass for ReLU activation

# In[65]:



def relu_forward(x):
    cache = x
    relu_output = np.maximum(0, x)
    return relu_output, cache


# # Checking Naive Forward Pass for ReLU activation

# In[66]:



x_shape = (2, 9)
x = np.random.randint(-9, 9, x_shape)
result, cache = relu_forward(x)
print(cache)
print(result)


# # Backward Pass for ReLU activation

# In[67]:





def relu_backward(derivatives_out, cache):
    x = cache
    temp = x > 0
    dx = temp * derivatives_out
    return dx


# **Adam Optimizer**

# In[68]:



def adam(w, dw, config=None):

    if config is None:
        config = {}

    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)

    config['t'] += 1
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dw
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * (dw**2)

    mt = config['m'] / (1 - config['beta1']**config['t'])
    vt = config['v'] / (1 - config['beta2']**config['t'])
    next_w = w - config['learning_rate'] * mt / (np.sqrt(vt) + config['epsilon'])
    return next_w, config
 


# # Testing Backward Pass for ReLU activation

# In[69]:



x_shape = (2, 9)
derivatives_out_shape = (2, 9)
x = np.random.randint(-9, 9, x_shape)
derivatives_out = np.random.randint(-9, 9, derivatives_out_shape)
result, cache = relu_forward(x)
dx = relu_backward(derivatives_out, cache)


print('Input x:\n', cache)
print('\nUpstream derivatives:\n', derivatives_out)
print('\nGradient with respect to x:\n', dx)


# # Softmax Classification Loss

# In[70]:




def softmax_loss(x, y):
    # Calculating probabilities
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probabilities = shifted_logits - np.log(z)
    probabilities = np.exp(log_probabilities)

    # Getting number of samples
    N = x.shape[0]

    # Calculating Logarithmic Loss
    loss = -np.sum(log_probabilities[np.arange(N), y]) / N

    # Calculating gradient
    dx = probabilities
    dx[np.arange(N), y] -= 1
    dx /= N

    # Returning tuple of Logarithmic loss and gradient
    return loss, dx


# # Creating Convolutional Neural Network Model

# In[71]:






class ConvNet1(object):



    def __init__(self, input_dimension=(3, 32, 32), number_of_filters=32, size_of_filter=7,
                 hidden_dimension=100, number_of_classes=10, weight_scale=1e-3, regularization=0.0,
                 dtype=np.float32):
      
        self.params = {}

        self.regularization = regularization

        self.dtype = dtype
 
        C, H, W = input_dimension

        HH = WW = size_of_filter

        F = number_of_filters

        Hh = hidden_dimension

        Hclass = number_of_classes

        self.params['w1'] = weight_scale * np.random.rand(F, C, HH, WW)
        self.params['b1'] = np.zeros(F)



        self.cnn_params = {'stride': 1, 'pad': int((size_of_filter - 1) / 2)}
        Hc = int(1 + (H + 2 * self.cnn_params['pad'] - HH) / self.cnn_params['stride'])
        Wc = int(1 + (W + 2 * self.cnn_params['pad'] - WW) / self.cnn_params['stride'])



        self.pooling_params = {'pooling_height': 2, 'pooling_width': 2, 'stride': 2}
        Hp = int(1 + (Hc - self.pooling_params['pooling_height']) / self.pooling_params['stride'])
        Wp = int(1 + (Wc - self.pooling_params['pooling_width']) / self.pooling_params['stride'])


        self.params['w2'] = weight_scale * np.random.rand(F * Hp * Wp, Hh)
        self.params['b2'] = np.zeros(Hh)



        self.params['w3'] = weight_scale * np.random.rand(Hh, Hclass)
        self.params['b3'] = np.zeros(Hclass)


        for d_key, d_value in self.params.items():
            self.params[d_key] = d_value.astype(dtype)


    def loss_for_training(self, x, y):
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        w3, b3 = self.params['w3'], self.params['b3']

        # Implementing forward pass for ConvNet1 and computing scores for every input
        # Forward pass:
        # Input --> Conv --> ReLU --> Pool --> FC --> ReLU --> FC --> Softmax
        cnn_output, cache_cnn = cnn_forward_pass(x, w1, b1, self.cnn_params)
        relu_output_1, cache_relu_1 = relu_forward(cnn_output)
        pooling_output, cache_pooling = max_pooling_forward_pass(relu_output_1, self.pooling_params)
        fc_hidden, cache_fc_hidden = fc_forward(pooling_output, w2, b2)
        relu_output_2, cache_relu_2 = relu_forward(fc_hidden)
        scores, cache_fc_output = fc_forward(relu_output_2, w3, b3)

        # Computing loss and gradients
        loss, d_scores = softmax_loss(scores, y)

        # Adding L2 regularization
        loss += 0.5 * self.regularization * np.sum(np.square(w1))
        loss += 0.5 * self.regularization * np.sum(np.square(w2))
        loss += 0.5 * self.regularization * np.sum(np.square(w3))

        # Implementing backward pass for ConvNet1
        # Backward pass through FC output
        dx3, dw3, db3 = fc_backward(d_scores, cache_fc_output)
        # Adding L2 regularization
        dw3 += self.regularization * w3

        # Backward pass through ReLU and FC Hidden
        d_relu_2 = relu_backward(dx3, cache_relu_2)
        dx2, dw2, db2 = fc_backward(d_relu_2, cache_fc_hidden)
        # Adding L2 regularization
        dw2 += self.regularization * w2

        # Backward pass through Pool, ReLU and Conv
        d_pooling = max_pooling_backward_pass(dx2, cache_pooling)
        d_relu_1 = relu_backward(d_pooling, cache_relu_1)
        dx1, dw1, db1 = cnn_backward_pass(d_relu_1, cache_cnn)
        # Adding L2 regularization
        dw1 += self.regularization * w1

        # Putting resulted derivatives into gradient dictionary
        gradients = dict()
        gradients['w1'] = dw1
        gradients['b1'] = db1
        gradients['w2'] = dw2
        gradients['b2'] = db2
        gradients['w3'] = dw3
        gradients['b3'] = db3

        # Returning loss and gradients
        return loss, gradients

    # Defining function for calculating Scores for Predicting.
    def scores_for_predicting(self, x):
        # Getting weights and biases
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        w3, b3 = self.params['w3'], self.params['b3']

        # Implementing forward pass for ConvNet1 and computing scores for every input
        # Forward pass:
        # Input --> Conv --> ReLU --> Pool --> FC --> ReLU --> FC --> Softmax
        cnn_output, _ = cnn_forward_pass(x, w1, b1, self.cnn_params)
        relu_output_1, _ = relu_forward(cnn_output)
        pooling_output, _ = max_pooling_forward_pass(relu_output_1, self.pooling_params)
        affine_hidden, _ = fc_forward(pooling_output, w2, b2)
        relu_output_2, _ = relu_forward(affine_hidden)
        scores, _ = fc_forward(relu_output_2, w3, b3)

        # Returning scores for every input
        return scores


# # Initializing new Model and checking dimensions of weights for every Layer

# In[72]:



model = ConvNet1(hidden_dimension=500)

N = 5
x = np.random.randn(N, 3, 32, 32)  
y = np.random.randint(10, size=N)  


loss, gradients = model.loss_for_training(x, y)


for param_name in model.params:
    print(param_name, model.params[param_name].shape)  
    print(param_name, gradients[param_name].shape)  
    print()

