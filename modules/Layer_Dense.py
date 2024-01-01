
import numpy as np

class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from input ones, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases
    # Backward pass
    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)



'''
- this class defines a dense (fully connected) layer of the network

- the init method takes in the number of input that the layer is going to be receiving and defines the number of neurons that the layer will contain
    - it initializes with random weights for each connection between the previous layer and the neurons
    - it initializes zeros for the biases

- the forward method implements a forward pass through the layer
    - it takes in the input values and:
        1. stores a copy of the inputs
        2. calculates the weighted sum of inputs using the dot product with weights
        3. adds the biases
        4. stores the result in the output

- the backward method implements a backward pass through the layer
    - this is used in training to update the weights and biases
    
    I NEED TO FIGURE THIS OUT


'''