"""
Task 01 : Coding Task 01: Implement forward and backpropagation for Figure 01 Neural Network. 
Use ReLU in the hidden layer and sigmoid in the output layer as an activation function. 
(Using Figure 01)
"""

import numpy as np
import math

################### Activation Functions ####################
## ReLu Activation Function
def ReLu(z):
    if z > 0:
        return z
    else:
        return 0

## Sigmoid Activation Function
def sigmoid(z):
    return (1/(1 + math.exp(-z)))

## Derivative of relu activation
def reluDerivative(x):
    """
    Finding the gradient of relu activation function
    """
    x[x<=0] = 0
    x[x>0] = 1
    return x


################# Forward Propagation ##################
def forward(X, W1, W2):
    """
    Arguments:
        X = inputs
        W1 = weights of hidden layer
        W2 = weights of output layer
        
    Returns:
        Y = Predicted output
    """
    ## Forward Pass
    Z1 = np.dot(X, W1.T)
    A1 = []
    for i in Z1.T:
        #print(i)
        A1.append(ReLu(i))
    A1 = np.array(A1).reshape(4, 1)
    
    Z2 = np.dot(W2, A1)
    Y_hat = sigmoid(Z2)
    
    return Y_hat


############### Loss Fucntion  ################
def cost_func(Y, X, Y_hat):
    """
    Arguments:
        Y : Real output
        Y_hat : Predicted output
        
    Returns:
        loss : Binary Crossentropy Loss
    """
    m = X.shape[0]
    loss = -1/m*(np.sum(Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat)))
    return loss
    



################## Backward propagation ######################
def backprop(loss, X, W1, W2, Y, Y_hat):
    """
    Arguments :
            loss : Model loss
            X : inputs
            W1 : hidden layer weights
            W2 : output layer weights
            Y : Real output
            Y_hat : Predicted output
            
    Returns :
            dw1 : gradient of loss w.r.t. W1
            dw2 : gradient of loss w.r.t. W2
    """
    
    m = X.shape[0]
    dw2 = 1/m*(np.dot(W2, ((Y-Y_hat).T)))
    ## print(dw2)
    dw1 = reluDerivative(W1)
    
    return [dw1, dw2]


################# Trainig of Neural Network  ######################
def train(epochs, lr, X, W1, W2, Y):
    """
    Arguments :
            epochs : No. of Epochs
            lr : learning rate
            X : inputs
            W1 : hidden layer weights
            W2 : output layer weights
            Y : Real output
            Y_hat : Predicted output
            
    Returns :
            W1 : Updated W1
            W2 : Updated W2
    """
    for i in range(epochs):
        Y_hat = forward(X, W1, W2)
        #print('Forward Propagation Completed for epoch  =  ' + str(i+1))
        loss = cost_func(Y, X, Y_hat)
        print('Loss for epoch  ' + str(i+1) + ' =  ' + str(loss))
        grad = backprop(loss, X, W1, W2, Y, Y_hat)
        dw1 = grad[0]
        dw2 = grad[1]
        W1 = W1 - lr*dw1
        W2 = W2 - lr*dw2
    return [W1, W2]


#################### Prediction of the network #####################
def prediction(X, W1, W2):
    """
    Arguments:
        X = inputs
        W1 = weights of hidden layer
        W2 = weights of output layer
        
    Returns:
        Y = Predicted output
    """
    Z1 = np.dot(X, W1.T)
    A1 = []
    for i in Z1.T:
        #print(i)
        A1.append(ReLu(i))
    A1 = np.array(A1).reshape(4, 1)
    
    Z2 = np.dot(W2, A1)
    Y_hat = sigmoid(Z2)
    
    return Y_hat


################ Model traing & Inference ##################
## Inputs and Weights
X = np.random.randn(1, 2)
W1 = np.random.randn(4, 2)
W2 = np.random.randn(1, 4)

epochs = 5
lr = 0.01
Y = np.array(1) ## Predicting actual output as 1
updated_weights = train(epochs, lr, X, W1, W2, Y)

W1 = updated_weights[0]
w2 = updated_weights[1]

output = prediction(X, W1, W2)
print('\n\nPredicted Output', output)
