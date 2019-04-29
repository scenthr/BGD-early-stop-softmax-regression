#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:52:55 2019

@author: spavko
"""

## Load the Iris dataset

import seaborn as sns
import numpy as np

iris = sns.load_dataset('iris')

X = iris.drop(['sepal_length','sepal_width','species'],axis=1).values
y = iris.species


# Add the 1s for the first weight (coefficient)
X_with_bias = np.c_[np.ones([len(X),1]),X]

# We set the random seed so that we can reproduce the results
np.random.seed(42)

# Split the dataset into train, validation and test datasets
def split_data(features,response,test_ratio=0.2,val_ratio=0.2):
    
    total = len(features)
    test = int(total * test_ratio)
    val = int(total * val_ratio)
    train = total - test - val
    
    random_data = np.random.permutation(total)
    
    X_train = features[random_data[:train]]
    y_train = response[random_data[:train]]
    X_valid = features[random_data[train:-test]]
    y_valid = response[random_data[train:-test]]
    X_test = features[random_data[-test:]]
    y_test = response[random_data[-test:]]
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

X_train, y_train, X_valid, y_valid, X_test, y_test = split_data(X_with_bias,y)

# Now we transform the response variable values to a vector o values where
# 1 in the vector represents the current value of the class

string_to_int = dict((s,i) for i, s in enumerate(y.unique()))

def transform(y, string_to_int):
    
    y_i = [string_to_int[s] for s in y]
    n = len(string_to_int)
    m = len(y)
    vector = np.zeros([m,n])
    vector[np.arange(m),y_i] = 1

    return vector

# transform the response variable to a vector
y_train_one_hot = transform(y_train, string_to_int)
y_valid_one_hot = transform(y_valid, string_to_int)
y_test_one_hot = transform(y_test, string_to_int)

# y_train_old = y_train
# y_valid_old = y_valid
# y_test_old = y_test

# y_train = y_train_old
# y_test = y_test_old
# y_valid = y_valid_old

# Transform also the original values of the categories to integers for easier
# comparison

y_train = [string_to_int[s] for s in y_train]
y_valid = [string_to_int[s] for s in y_valid]
y_test = [string_to_int[s] for s in y_test]
 
print('X_train: ', X_train.shape)
print('X_valid: ', X_valid.shape)
print('X_test: ', X_test.shape)

print('y_train_one_hot: ', y_train_one_hot.shape)
print('y_valid_one_hot: ', y_valid_one_hot.shape)
print('y_test_one_hot: ', y_test_one_hot.shape)

# Now we define a Softmax function as
def softmax(logits):
    exps = np.exp(logits)
    exp_sums = np.sum(exps, axis=1, keepdims=True)
    return exps / exp_sums

n_inputs = X_train.shape[1]
n_outputs = len(np.unique(y_train))
    
print('X_train.shape: ', X_train.shape)
print('n_inputs: ',n_inputs)
print('n_outputs: ',n_outputs)

# Training the data that we have prepared so far

eta = 0.1  # set to 0.01 if you want a slower learning rate
n_iterations = 20001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1     #regularization hyperparameter 
best_loss = np.infty  

# Randomize the parameters initially
Theta = np.random.randn(n_inputs, n_outputs)    
    
for iteration in range(n_iterations):
    
    logits = X_train.dot(Theta)    
    y_proba = softmax(logits)
    xentropy_loss = -np.mean(np.sum(y_train_one_hot * np.log(y_proba + epsilon),axis=1))
    l2_loss = 1/2 * np.sum(np.square(Theta[1:]))  ## everything except the 
                                                  ## first weight
    error = y_proba - y_train_one_hot
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, n_outputs]), 
                                   alpha * Theta[1:]]
    Theta = Theta - eta * gradients

    logits = X_valid.dot(Theta)
    y_proba = softmax(logits)
    xentropy_loss = -np.mean(np.sum(y_valid_one_hot * np.log(y_proba + epsilon), axis=1))
    l2_loss = 1/2 * np.sum(np.square(Theta[1:]))
    loss = xentropy_loss + alpha * l2_loss
    if iteration % 500 == 0:
        print(iteration, loss)
    if loss < best_loss:
        best_loss = loss
    else:
        print(iteration - 1, best_loss)
        print(iteration, loss, "early stopping!")
        break

# trained model parameters
Theta    
    
# and now we can make predictions, and check out the accuracy
logits = X_valid.dot(Theta)
y_proba = softmax(logits)
y_predict = np.argmax(y_proba, axis=1)

accuracy = np.mean(y_predict == y_valid)
print('Accuracy: ',accuracy)
    
## Pretty good accuracy

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    