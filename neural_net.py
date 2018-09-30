import numpy as np
import os
import csv

## Fully Connected layer


class FullyConnectedLayer(object):
    def __init__(self, num_input, num_output, lr=1e-3, scale=2, weights=[], bias=[]):
        self.num_input = num_input
        self.num_output = num_output
        self.incoming_weights = np.array(weights).astype(np.float32)
        self.input_data = np.empty((0))
        self.lr = lr
        self.scale = scale
        self.bias = np.array(bias).astype(np.float32)

    def forward(self, input_data):
        self.input_data = input_data
        outputs = np.dot(input_data, self.incoming_weights) + self.bias
        return outputs

    def backward(self, gradient_data):
        # index i: input nodes; j: output nodes
        gradients = np.dot(gradient_data, np.transpose(self.incoming_weights))

        # do updates on the weights (*np.dot = np.matmul)
        updates = np.multiply(self.lr,
                                             np.dot(
                                                 np.transpose(self.input_data),
                                                         gradient_data)
                                             )
        self.incoming_weights -= updates

        # update bias
        self.bias -= np.multiply(self.lr, np.sum(gradient_data, axis=0))  # just sum, since we already /num_samples before
        return gradients


## ReLU layer
class ReLULayer(object):
    def __init__(self):
        self.input_data = []

    def forward(self, input_data):
        self.input_data = input_data
        input_data = np.clip(input_data, 0, input_data)
        return input_data


    '''
    Input: gradient_data; size = num_samples x num_output_nodes = num_samples x num_input_nodes (**)
    input_data; size = num_samples x num_input_nodes
    
    ** num_input_nodes = num_output_nodes
    
    Output: gradient_data; size = num_samples x num_input_nodes
    '''
    def backward(self, gradient_data):
        # do element-wise multiplication
        gradient_data = np.multiply(gradient_data, self.input_data >= 0)
        return gradient_data


## Output layer
class SoftmaxOutput_CrossEntropyLossLayer(object):
    def __init__(self):
        self.loss = np.empty((0))
        self.input_data = np.empty((0))
        self.label_data = np.empty((0))


    '''
    X: the output of the last fully-connected layer -- each row in X represents a training sample
    i-th row, j-th column of X represents the value of the j-th output node, for the i-th training sample
    '''
    def softmax(self, X):
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)


    '''
    Input: X = the output of the last fully-connected layer; size = num_training_samples x number of output classes
    y = output labels (correct classes); size = num_training_samples x 1 (the single entry in each row is the correct output   
    class' index)
    
    Output: cross-entropy loss = ( \sum_{t = 1}^{num_training_samples} ( \sum_{i = 1}^{num_classes} y_i * \log(p_i) )) /
    num_training_samples
    '''
    def cross_entropy(self, X, y):
        num_training_samples = y.shape[0]
        softmax_result = self.softmax(X)

        '''Here, for each t-th row in softmax_result, the i-th entry represents the probability associated with the i-th output
        class. Let p = softmax_result. If we consider y = [0,...,0,1,0,...,0], where y_i = 1 only at the correct output class k,
        then \sum_{i = 1}^{num_classes} y_i * \log(p_i) = y_k(=1) * \log(p_k), since all other components of the sum = 0.
        Multiplying the true label index k of the t-th sample, y_{t, k}, to \log(p_{t, k}) is equivalent to getting the
        probability associated with the y_i-th output class. 
        '''
        log_likelihood = -np.log(softmax_result[range(num_training_samples), y] + 1e-6)

        self.loss = np.sum(log_likelihood) / num_training_samples
        return self.loss


    '''
    Input: X = the output of the last fully-connected layer; size = num_training_samples x number of output classes
    y = output labels (correct classes); size = num_training_samples x 1 (the single entry in each row is the correct output
    class' index)

    Output: gradients (for each output node) to be propagated back to the FC layer. For the t-th sample (t-th row), for the
    true class k, this is equivalent to p_{t, k} - y_{t, k} = p_{t, k} - 1. For the rest of the classes c, the gradient is
    p_{t, c} - y_{t, c} = p_{t, c} - 0
    '''
    def delta_cross_entropy(self, X, y):
        num_training_samples = y.shape[0]
        softmax_result = self.softmax(X)
        softmax_result[range(num_training_samples), y] -= 1  # because y_{t, k} = 1; so p_{t, k} - y_{t, k} = p_{t,k} - 1
        softmax_result /= num_training_samples  # divide beforehand, can sum the rows up later.
        return softmax_result

    def eval(self, input_data, label_data, phase):
        self.input_data = input_data
        self.label_data = label_data
        if phase == 'train':
            self.loss = self.cross_entropy(input_data, label_data)

    def backward(self):
        gradient_data = self.delta_cross_entropy(self.input_data, self.label_data)
        return gradient_data


def network_forward(network, input_data, label_data=None, phase='train'):
    for layer in network:
        if type(layer) is not SoftmaxOutput_CrossEntropyLossLayer:
            input_data = layer.forward(input_data)
        else:
            layer.eval(input_data, label_data, phase)
    return network


def network_backward(network):
    for layer in reversed(network):
        if type(layer) is SoftmaxOutput_CrossEntropyLossLayer:
            gradient = layer.backward()
        else:
            gradient = layer.backward(gradient)
    return network


## helper functions to read data
def read_data(file_name):
    data_set = []
    with open(file_name, newline='') as csvfile:
        data_file = csv.reader(csvfile)
        for row in data_file:
            data_set.append([float(x) for x in list(row)[1:]])
    return data_set

# set current working directory
dir = os.getcwd()

def main():
    pass

if __name__ == '__main__':
    main()
