import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def init_network():
  network={}
  network['w1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
  network['b1'] = np.array([0.1, 0.2, 0.3])
  network['w2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
  network['b2'] = np.array([0.1, 0.2])
  network['w3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
  network['b3'] = np.array([0.1, 0.2])
  return network
  

def forward(network, x):
    W1 = network['w1']
    W2 = network['w2']
    W3 = network['w3']

    b1 = network['b1']
    b2 = network['b2']
    b3 = network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

def identity_function(x):
    return x

def softmax(a):
    exp_a=np.exp(a)
    sum_exp_a=np.sum(exp_a)
    y= exp_a/ sum_exp_a
    return y

a=np.arry([0.3, 2.9, 4.0])

y = softmax(a)

print(y)


   








