# NN-Training-as-Dynamical-Process

MATLAB scripts for computing “Neural Network Training as a Dissipative Process”
Authors: Marco Gori, Marco Maggini, Alessandro Rossi (2016)
Contacts: rossi111@unisi.it

makeSystemMatrix.m : computes the matrices of the dynamical system

PlotImpulsiveResponse.m : plots the Impulsive Response of the dynamical system, given the roots of the characteristic polynomial

TRnet.m : define a matlab object for the implementation of a 2-layer Neural Network, trainable in a standard mode or with the dissipative dynamic system proposed in the paper; data must be provided in a single variable of type 'struct' with field:
  - X : input_size-by-number_of_samples matrix of training data;
  - Y : output_size-by-number_of_samples matrix of target for training data (Inf/NaN means unsupervised sample);
  - Xtest : input_size-by-number_of_samples matrix of test data;
  - Ytest : output_size-by-number_of_samples matrix of target for test data(Inf/NaN means unsupervised sample); 
Quick start commands - once you define your data, say in a variable 'Data' and choose the max number of training epochs: 
    net = TRnet;net.train(data,maxEpochs);

reducedMnistData.mat : contains a subset (10000 elements) of the MNIST dataset [1]

vowels.mat : contains a toy dataset for vowels classification task (many unsupervised samples, target = Inf)


[1] see: http://yann.lecun.com/exdb/mnist/
