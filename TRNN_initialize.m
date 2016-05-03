function net = TRNN_initialize(InputDim,OutputDim,Units,parameters)
% TRNN_initialize : initialization of the parameters for a 2-layer NN, one 
%                 hidden layer, rectifier as activation function, identity 
%                 as output function, max function for class prediction on 
%                 binary target; the weights(neurons) will be optimized 
%                 following the Time-Regularization algorithm
%
%     net = TRNN_initialize(InputDim,OutputDim,Units,parameters)
%
%     InputDim: input (features) space dimension
%     OutputDim: output dimension, number of classes of data
%     Units: number of units in the hidden layer
%     parameters: vector of parameters of the system of differential
%                 equations(see BuildingMatrix for further explanations)
%
% Author: Alessandro Rossi (2016)
%         rossi111@unisi.it
    
net.MSE = [];% vectors containing the MSE evaluation at each step
net.TestMSE = [];   
net.Accuracy = [];% vectors containing the Accuracy evaluation at each step
net.TestAccuracy = [];
net.Epochs = 0;% keep tracks of the training epochs of the system
net.InputDim = InputDim;
net.OutputDim = OutputDim;

[net.A,net.B,net.theta,net.order,net.solutions]=BuildingMatrix(parameters);

net.Units = Units ;        
net.HiNe = net.Units *(InputDim+1); % calculating the number of neurons in
                                    % the hidden layer
net.OutNe = (net.Units+1)*OutputDim; % calculating the  number of neurons
                                     % in the output layer
       
% initialization of the hidden and output matrix weights (neuirons), each 
% column contains the Cauchy's Initial Condition of the differential 
% equations of a weight,the initial value can be chosen randomly thanks to 
% the asymptotic stability, this allows us to differentiate the weights, 
% the derivatives start from 0
net.W = zeros(net.order,net.HiNe); % hidden neurons matrix
net.C = zeros(net.order,net.OutNe); % output neurons matrix   
% random initialization of neurons
net.W(1,:)=rand(1,net.Units*(InputDim+1))*1e-3;
net.C(1,:)=rand(1,(net.Units+1)*OutputDim)*1e-3;
% null mean setting of weights
net.W(1,:)=net.W(1,:)-mean(net.W(1,:)); 
net.C(1,:)=net.C(1,:)-mean(net.C(1,:));
% organising neurons in matrix form
net.HM = zeros(net.Units,InputDim+1);
net.OM = zeros(OutputDim,net.Units+1);
net.HM(1:net.HiNe)=net.W(1:net.order:(net.HiNe*net.order));
net.OM(1:net.OutNe)=net.C(1:net.order:(net.OutNe*net.order));

% zero-initialization of the penalty function gradient wrt the weights
net.V_w = zeros(size(net.HM));
net.V_c = zeros(size(net.OM));

end
