
import numpy as np

class Activation_Softmax:
    # Forward Pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        # Output the probabilities
        self.output = probabilities
    # Backward pass
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


'''
 - a softmax activation function allows our model to act as a 'classifier'
 - this type of activation function provides context to the final outputs, unlike functions that output contextless numbers
 - a softmax activation function produces a 'normalized' distribution of probabilities as the model's output
    - the outputs essentially become confidence scores
    - each value from our model has passed through the softmax activation function and therefore represents the probability of the output value being 'correct'
'''