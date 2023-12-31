
import numpy as np

class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)
    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify the original variable, let's make a copy of the values first
        self.dinputs = dvalues.copy()
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


'''
- this class defines a Rectified Liner Unit activation function
- for any input value that is less than zero, function will output zero
- for any input value that is greater than zero, function will output the input value

- forward method calls the activation function on the inputs
    - applies the ReLU to the inputs
    - uses maximum to change any negative numbers to a zero
    - stores the input and the outputs

- backward method computes the gradient of the loss with respect to the input
    - takes in the gradient of the loss with respect to the output (dvalues)
        - the gradient is the rate of change of a quantity with respect to its inputs
        - gradient is used to update the model's parameters during optimization
    - sets the gradient to 0 where the corresponding input values were negative during the forward pass

'''



# example of how to use the forward method
'''
sample_activation_one = Activation_ReLU()
values = [-2, -1, 0, 1, 2, 3, 4, 100]
for value in values:
    sample_activation_one.forward(value)
    print(sample_activation_one.output)
'''



# example of how to use the backward method
'''
sample_activation_two = Activation_ReLU()
values = np.array([-2, -1, 0, 1, 2, 3, 4, 100], dtype=float)
sample_activation_two.forward(values)

print("Output after forward pass:")
print(sample_activation_two.output)

# Assume you have a loss gradient with respect to the output (dvalues)
loss_gradient = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=float)

# Perform the backward pass
sample_activation_two.backward(loss_gradient)

# Print the gradient with respect to the inputs after the backward pass
print("Gradient with respect to the inputs after backward pass:")
print(sample_activation_two.dinputs)
'''