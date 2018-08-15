'''
Implemented a simple MLP network in order to predict stock market trends.
This network uses three hidden layers and a Sigmoid activation function 
to successfully train the inputed closing data. The hidden layers contain
128 nodes, 64 nodes, and 8 nodes while using the k-fold cross validation to
train the data

By: Michael Rzadki
'''

#Setting up the required imports and extensions
#that will be used to construct this ANN
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.model_selection import KFold
import matplotlib
import matplotlib.pyplot as plt
from dateutil.parser import parse
from datetime import datetime, timedelta
from collections import deque

#Opening the AIGL data set 
dataframe = pd.read_csv('ETFS Livestock Data.csv')
dataframe.describe()
dataframe.tail(10)

#Selecting the correct data and eliminating unwanted info
desired_columns = ['Open', 'Last Close']
basic_mlp_data = dataframe[desired_columns]
basic_mlp_data.head()

# Setting up the Tensorflow graph
tf.reset_default_graph()

# Defining the Neural Network topology with the hidden layer sizes
#as well as the amount and epochs and how quickly the ANN will learn.
amountEpochs = 1000
batchSize = 128
netHiddenSizes = [128, 64, 8]
learningRate = 0.001
strength = 0.01
nonLinearity = tf.nn.relu
dropoutAmount = 0.7

#The input for the graph will be the open price and the target 
#will consist of the Last Close price. A placeholder will also
#be used to pass a dropout rate 
netInput = tf.placeholder(tf.float32, shape= [None, 1])
netTarget = tf.placeholder(tf.float32, shape= [None, 1])
dropoutProb = tf.placeholder(tf.float32)

# Uses an L2 to regulate the weights in order to prevent them from 
#growing too quickly
regulariser = tf.contrib.layers.l2_regularizer(scale=strength)

# Building the network from a variety of different dimensions
#Uses a nonLinearity (Sigmoid function)as the activation along
#with a kernel regularizer
net = netInput
for size in netHiddenSizes:
    net = tf.layers.dense(inputs = net, 
                          units = size, 
                          activation = nonLinearity, 
                          kernel_regularizer = regulariser)
   
    net = tf.layers.dropout(inputs = net,
                            rate = dropoutProb)

#This simple MLP will produce a linear output value  
netOutput = tf.layers.dense(inputs = net,
                            units = 1, 
                            activation = None, 
                            kernel_regularizer = regulariser)    

#The loss that is determined for punishing the network
#based on how efficient it is (MSE)
loss = tf.losses.mean_squared_error(labels = netTarget, 
                                    predictions = netOutput)

#Applying a L2_loss to the current loss using Tensorflow
l2Variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
l2Loss = tf.contrib.layers.apply_regularization(regulariser, 
                                                 l2Variables)
totalLoss = loss + l2Loss

#Starts the training of the ANN with the initialzation of the 
#Tensorflow operations that will be required 
trainOp = tf.train.AdamOptimizer(learningRate).minimize(totalLoss)
initOp = tf.global_variables_initializer()

with tf.Session() as sess:
    
#Sets up the amount of folds that will be used as well as 
#the fold error and iteration 
    amountFolds = 5
    kFolds = KFold(n_splits=amountFolds)
    data = basic_mlp_data.as_matrix()
    foldIteration = 0
    foldErrors = []
    
#Uses K-fold to cross validate the given dataset
for trainIndices, test_indices in kFolds.split(data):

        #After each new fold, the network is reinitialized 
        sess.run(initOp)
        
        #This starts the training phase of the MLP
        for epoch in range(amountEpochs):
            
            #Each epoch, result in the training set being switched
            randomTrainIndices = np.random.permutation(trainIndices)
            trainSet = data[randomTrainIndices]
            
            #Starts to loop the training set in order to help
            #optimize the network
            for begin in range(0, len(trainSet), batchSize):
                end = begin + batchSize
                batch_x = trainSet[begin:end].T[0].reshape((-1, 1))
                batch_y = trainSet[begin:end].T[1].reshape((-1, 1))
                
                sess.run(trainOp, feed_dict={
                    netInput: batch_x,
                    netTarget: batch_y,
                    dropoutProb: dropoutAmount
                })
        
        #This starts the testing phase of the MLP
        testSet = data[test_indices]
        
        #Determines the error found when completing the test set
        allError = []
        for begin in range(0, len(testSet), batchSize):
            end = begin + batchSize 
            batch_x = trainSet[begin:end].T[0].reshape((-1, 1))
            batch_y = trainSet[begin:end].T[1].reshape((-1, 1))
            
            error = sess.run(loss, feed_dict={
                netInput: batch_x,
                netTarget: batch_y,
                dropoutProb: 1.0
            }) 
            allError.append(error)
        
        allError = np.array(allError).reshape((-1))
        foldErrors.append(allError)
    
        
        #Displays the Error mean (MSE)and error deviation 
        print("\nFold iteration:  ", foldIteration,
              "\nMSE:             ", np.mean(allError),
              "\nError deviation: ", np.std(allError),
              "\n")
        foldIteration += 1      
        
fold_errors = np.array(foldErrors).reshape((amountFolds, -1))
    
hist_data = dict()
keys = ['fold 0', 'fold 1', 'fold 2', 'fold 3', 'fold 4']
for i, key in enumerate(keys):
    hist_data[key] = fold_errors[i]
     


