# import sys
# import numpy as np
# import matplotlib
# import nnfs
# import matplotlib.pyplot as plt



# inputs = [1, 2, 3, 2.5]

# weights1 = [0.2, 0.8, -0.5, 1.0]
# weights2 = [0.5, -0.91, 0.26, -0.5]
# weights3 = [-0.26, -0.27, 0.17, 0.87]

# bias1 = 2
# bias2 = 3
# bias3 = 0.5

# output = [inputs[0]*weights1[0]+inputs[1]*weights1[1]+inputs[2]*weights1[2]+inputs[3]*weights1[3]+bias1,
#     inputs[0]*weights2[0]+inputs[1]*weights2[1]+inputs[2]*weights2[2]+inputs[3]*weights2[3]+bias2,
#     inputs[0]*weights3[0]+inputs[1]*weights3[1]+inputs[2]*weights3[2]+inputs[3]*weights3[3]+bias3]
# print(output)


# ########################################


# inputs = [1, 2, 3, 2.5]

# weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]

# biases = [2, 3, 0.5]

# layer_outputs = []
# for neuron_weights, neuron_bias in zip(weights, biases):
    # neuron_output = 0
    # for n_input, weight in zip(inputs, neuron_weights):
    #         neuron_output += n_input*weight
    # neuron_output += neuron_bias
    # layer_outputs.append(neuron_output)
# print(layer_outputs)


# ########################################


# test_inputs = [1,2,3,2.5] 
# # inputs are the values that the previous layer is outputting (these represent the 'nodes')
# test_weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
# # weights are the values in which we multiply the incoming data (these represent the 'lines' that connect the previous nodes to the ones that we are modeling)
# bias = [2,3,0.5]
# output = np.dot(test_weights, test_inputs) + biases


# BATCHING inputs
# test_inputs = [[1,2,3,2.5],
#     [2.0, 5.0, -1.0, 2.0],
#     [-1.5, 2.7, 3.3, -0.8]] 

# test_weights = [[0.2, 0.8, -0.5, 1.0],
#     [0.5, -0.91, 0.26, -0.5],
#     [-0.26, -0.27, 0.17, 0.87]]

# biases = [2, 3, 0.5]

# test_weights2 = [[0.1, -0.14, 0.5],
#     [-0.5, 0.12, -0.33],
#     [-0.44, 0.73, -0.13]]

# biases2 = [-1, 2, -0.5]

# layer1_output = np.dot(test_inputs, np.array(test_weights).T) + biases
# layer2_output = np.dot(layer1_output, np.array(test_weights2).T) + biases2
# print(layer2_output)


# ########################################


# # deep learning often uses X as the vaiable for the data set (inputs)
# # 3 samples here
# X = [[1,2,3,2.5],
#     [2.0, 5.0, -1.0, 2.0],
#     [-1.5, 2.7, 3.3, -0.8]]

# # if there is a trained model, that data would be used to initialize - otherwise we randomize before we have a trained model

# # np.random.seed(0)

# class Layer_Dense:
#     def __init__(self, n_inputs, n_neurons):
#         self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
#         self.biases = np.zeros((1, n_neurons))

#     def forward(self, inputs):
#         inputs = np.array(inputs)  # Convert inputs to a NumPy array
#         self.output = np.dot(inputs, self.weights) + self.biases

# layer1 = Layer_Dense(4, 10)
# layer2 = Layer_Dense(10, 10)
# layer3 = Layer_Dense(10, 10)
# layer4 = Layer_Dense(10, 2)

# layer1.forward(X)
# # print(layer1.output)
# layer2.forward(layer1.output)
# layer3.forward(layer2.output)
# layer4.forward(layer3.output)
# print(layer4.output)


# ########################################


# inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
# output = []

# for i in inputs:
#     if i > 0:
#         output.append(i)
#     elif i <= 0:
#         output.append(0)

# print(output)

import sys
import numpy as np
import matplotlib
import nnfs
import matplotlib.pyplot as plt

nnfs.init()

X = np.array([[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]])

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(4, 5)  # Adjust the number of inputs to match the number of features in your dataset
activation1 = Activation_ReLU()

layer1.forward(X)  # Convert X to a list and pass it to forward method

activation1.forward(layer1.output)

print(layer1.output)


# this function create the spiral dataset
# def spiral_data(points, classes):
#     X = np.zeros((points*classes, 2))
#     y = np.zeros(points*classes, dtype='uint8')
#     for class_number in range(classes):
#         ix = range(points*class_number, points*(class_number+1))
#         r = np.linspace(0.0, 1, points)  # radius
#         t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
#         X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
#         y[ix] = class_number
#     return X, y

# print('here')
# X, y = spiral_data(1000, 2)

# plt.scatter(X[:,0], X[:,1])
# plt.show()