
import numpy as np

class Loss:
    # Calculates the data and regularization losses given model output and ground truth values
    # output is the outout from the model
    # y is the intended target values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss


'''
loss is the difference between the predicted output and the actual target
loss functions take the predicted outputs of the model and the actual targets as inputs and compute a single value that indicates the dissimilarity between the predicted and actual values

this loss class return the mean loss for all of the samples
'''