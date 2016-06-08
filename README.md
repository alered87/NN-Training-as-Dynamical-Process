# NN-Training-as-Dynamical-Process

MATLAB scripts for computing “Neural Network Training as a Dissipative Process”
Authors: Marco Gori, Marco Maggini, Alessandro Rossi (2016)
Contacts: rossi111@unisi.it

makeSystemMatrix.m: compute the matrices of the dynamical system

PlotImpulsiveResponse.m: plot the Impulsive Response of the dynamical system, given the roots of the characteristic polynomial

TRnet.m: define a matlab object for the implementation of a 2-layer Neural Network, trainable in a standard mode or with the dissipative dynamic system proposed in the paper 

reducedMnistData.mat: contain a subset (10000 elements) of the MNIST(*) dataset
(*) see: http://yann.lecun.com/exdb/mnist/

Vowels.mat: contains a toy dataset for vowels classification task
