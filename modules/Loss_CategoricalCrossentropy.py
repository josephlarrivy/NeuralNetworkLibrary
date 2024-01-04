
from Common_Loss import Loss
import numpy as np

# inherits from Common_Loss class
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        # the clip here forces the values to be between 0.00001... and 0.99999...
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # Probabilities for target values - only if categorical labels
        # this is for when a user passes in scalar values
        if len(y_true.shape) == 1:
            # this will allow us to loop over every sample in the batch with range(samples)
            # for each 'row' / sample in the 2d array of y_clipped, we will grab the value at the index of y_true
            # EXAMPLE:
                # with y_pred_clipped = np.array([0.7, 0.2], [0.8, 0.1])
                # range(samples) will be [0, 1]
                # with y_true = [1, 0]
                # correct confidences will be [0.2, 0.8]
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            # this case uses an array of one-hot encoded vectors as y_true
            # we are again picking out from out y_pred the values that are at the indeces stated in y_true, but in this case, we need to porperly ready the one-hot encoded 2d array to get them
            # multiplication of every element of both 2d array occurs, but because the one-hot 2d array has zeros for the values that are not 'hot', when we add the results of the multiplication, we get only the values that were at the 'hot' points
            # EXAMPLE:
                # with y_pred_clipped = np.array([0.7, 0.2], [0.8, 0.1])
                # range(samples) will be [0, 1]
                # with y_true = [[1, 0], [1, 0]
                # correct confidences will be [0.7, 0.8]
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=1)

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        # final output in an array of losses
        # the final output will be an array of losses for each sample in the batch based on the true labels
        # the lower the value, the better the predictions align with the true labels
        # negative_log_likelihoods = [loss_for_sample_one_of_batch, loss_for_sample_two_of_batch, loss_for_sample_three_of_batch]
        return negative_log_likelihoods
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])
        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.dinputs = -y_true / dvalues
        # Normalize gradient
        self.dinputs = self.dinputs / samples


'''
loss helps us figure out how "wrong" our model is
this object class outputs the negative log of the predicted class's value

Lower loss values:
 - lower loss values mean that our model output more confdence in its prediction of the target class
 - this means that the predicted probability distribution aligns well with the actual class, and the model is making more accurate predictions

Higher Loss values:
 - higher loss values mean that our model output less confidence in its prediction of the target class
 - this suggests that the predicted probability distribution diverges from the actual class distribution, and the model may be struggling to correctly identify the target class

How loss works in optimization
 - during training, the goal is to minimize loss
 - optimization algorythums tweak the model's paramaters to improve the model's ability to correctly classify classes
    - this proces is done by working to lower the value of loss


'''