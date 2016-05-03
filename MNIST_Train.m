
% MNIST_Train : quickly implementation on the MNIST data set of a NN with
%               the Rectifier as actiavtion function and trained with the
%               Time-Regularization algorithm

load MNISTdata;
% setting system parameters
Theta = 1 ;
parameters = [Theta;4;1e8];
% High Dissipation Configuration
tau = 40 ;
eta = 1e-3;
% Low Dissipation Configuration
% tau = 1e-5 ;
% eta = 1e-8;

Units = 300 ;
InputDimension = 784; 
OutputDimension = 10 ;
iterations = 10 ;

net = TRNN_initialize(InputDimension,OutputDimension,Units,parameters);
net = TRNN_train(net,TrainRedux,TestRedux,eta,tau,iterations);

