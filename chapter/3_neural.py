import numpy as np
import matplotlib.pylab as plt

###############
# step function
###############
def step_function(x):
    return np.array(x > 0, dtype=int)

###############
# sigmoid function
###############
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

###############
# ReLU function
###############
def relu(x):
    return np.maximum(0, x)

# x = np.arange(-5.0, 5.0, 0.1)

# y = step_function(x) # step
# y = sigmoid(x)       # sigmoid
# y = relu(x)          # relu

# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)
# plt.show()

###############
# 각 층 신호 전달 구현
###############
def neural():
    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])                  # 가중치

    print(X.shape)  # (2,)
    print(W1.shape) # (2,3)
    print(B1.shape) # (3,)

    A1 = np.dot(X, W1) + B1
    Z1 = sigmoid(A1)

    print(A1)  # [0.3 0.7 1.1]
    print(Z1)  # [0.57444252 0.66818777 0.75026011]

    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])

    print(Z1.shape)
    print(W2.shape)
    print(B2.shape)

    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)

    print(A2)
    print(Z2)

    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])

    A3 = np.dot(Z2, W3) + B3
    Y = A3

    print(A3)
    print(Y)


# neural()

###############
# Softmax
###############

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


print(softmax([0.3, 2.9, 4.0]))
