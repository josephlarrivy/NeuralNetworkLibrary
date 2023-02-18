import sys
import numpy as np
import matplotlib




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


# deep learning often uses X as the vaiable for the data set (inputs)
# 3 samples here
X = [[1,2,3,2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

# if there is a trained model, that data would be used to initialize - otherwise we randomize before we have a trained model
class Layer_Dense:
    def __init__(self):
        pass
    def forward(self):
        pass