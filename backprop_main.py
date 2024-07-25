import matplotlib.pyplot as plt
import numpy as np
from backprop_network import *
from backprop_data import *

np.random.seed(0)  # For reproducibility

# Network configuration
layer_dims = [784, 40, 10]
net = Network(layer_dims)

#section b
n_train = 10000
n_test = 5000
x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)

# Training configuration
epochs = 30
batch_size = 10

learning_rates = [0.001, 0.01, 0.1, 1, 10]
fig, axs = plt.subplots(3)
fig.suptitle('Learning rate affects on test results')

for rate in learning_rates:
    _parameters, epoch_train_cost, _epoch_test_cost, epoch_train_acc, epoch_test_acc = net.train(
        x_train, y_train, epochs, batch_size, rate, x_test=x_test, y_test=y_test)
    axs[0].plot(epoch_train_cost, label=rate)
    axs[0].set_title("Training Loss")
    axs[1].plot(epoch_train_acc, label=rate)
    axs[1].set_title("Train Accuracy")
    axs[2].plot(epoch_test_acc, label=rate)
    axs[2].set_title("Test Accuracy")

plt.legend(loc="upper right")
plt.show()


# loading data for c + d
n_train = 60000
n_test = 10000
x_train, y_train, x_test, y_test = load_as_matrix_with_labels(n_train, n_test)
# section c


# Training configuration
epochs = 30
batch_size = 10
learning_rate = 0.1
_parameters, epoch_train_cost, _epoch_test_cost, epoch_train_acc, epoch_test_acc = net.train(
    x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)

print(epoch_test_acc[-1])

#section d
layer_dims = [784, 10]
net = Network(layer_dims)
epochs = 30
batch_size = 10
learning_rate = 0.1
parameters, _epoch_train_cost, _epoch_test_cost, epoch_train_acc, epoch_test_acc = net.train(
    x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)


plt.plot(epoch_train_acc, label="train accuracy")
plt.plot(epoch_test_acc, label="test accuracy")
plt.show()

# fig, axs = plt.subplots(1, 10)

# for i in range(10):
#     axs[i].imshow(np.reshape(parameters['W1'][i], (28, 28)), interpolation='nearest')

# plt.legend(loc="upper left")
# plt.show()

# section e
layer_dims = [784, 100, 40, 10]
net = Network(layer_dims)
epochs = 30
batch_size = 100
learning_rate = 0.1
parameters, _epoch_train_cost, _epoch_test_cost, epoch_train_acc, epoch_test_acc = net.train(
    x_train, y_train, epochs, batch_size, learning_rate, x_test=x_test, y_test=y_test)
