classdef TRnet < handle
    %  TRnet: neural network object for dissipative vs. standard training
    %  comparing

    properties
        parameters = [1;4;1e8]; % model parameters, see makeSystemMatrix
        tau = 0.01; % updatng time sampling step
        inputDim;
        outputDim;
        HU = 100; % # of hidden units
        weightsNormalization = 'global';% see weightsInitialization
        activation = 'rectifier'; % see transferFunction
        outputFunction = 'linear'; % see transferFunction
        eta = 1e-4; % learning rate
        etaDecay = 1; % learning rate decay factor per training epochs
        batchSize = 1;
        errStopThreshold = 1e-5; % minimum trheshold for Mean Square Error on training
        dissipativeTraining = 1; % boolean, 0 = classic training ; 1 = dissipative training
        gradientNormalization = 1; % gradient normalization (for batch mode)
        trainAccuracy = 0; % vector containing classification accuracy on training for each epoch
        trainMSE = 1; % vector containing MSE on training for each epoch
        testAccuracy = [];% vector containing classification accuracy on test for each epoch
        testMSE = [];% vector containing MSE on test for each epoch
        % computed
        A;B;theta;order;solutions;M;U; % Linear System Variables
        W = []; % hidden weights matrix
        O = []; % output weights matrix
        ah = []; % hidden linear neurons
        ao = []; % output states neurons
        z = []; % hidden activation
        o = []; % output activation
        dW = []; % hidden weights gradients matrix
        dO = []; % output weights gradients matrix
        HN = []; % dynamical system variables for hidden weights (order-by-n_of_hiddenweights matrix)
        ON = []; % dynamical system variables for output weights (order-by-n_of_outputweights matrix)
    end
    
    methods     
%% CONSTRUCTOR
        function net = TRnet(varargin)
            numberargs = nargin;
            if rem(numberargs,2) ~= 0
                error('Arguments must occur in name-value pairs.');
            end
            if numberargs > 0
                for i = 1:2:numberargs
                    if ~ischar(varargin{i})
                        error('Arguments name must be strings.');
                    end
                    net.(varargin{i}) = varargin{i+1};
                end
            end
        end

%% Setting Methods       
        function set.tau(net,tau)
            msg = 'tau must be a real number greater than 0';
            if ~isnumeric(tau) || tau<=0 || ~isfinite(tau) || isempty(tau)
                error(msg);
            else
                net.tau = tau;
            end
        end
%% 
        function set.inputDim(net,inputDim)
            msg = 'inputDim must be an integer (input space dimension)';
            if ((inputDim-round(inputDim))~=0)
                error(msg);
            else
                net.inputDim = inputDim;
            end
        end  
%% 
        function set.outputDim(net,outputDim)
            msg = 'outputDim must be an integer (output space dimension)';
            if ((outputDim-round(outputDim))~=0)
                error(msg);
            else
                net.outputDim = outputDim;
            end
        end
%% 
        function set.HU(net,HU)
            msg = 'HU (number of hidden units) must be an integer';
            if ((HU-round(HU))~=0)
                error(msg);
            else
                net.HU = HU;
            end
        end
%% 
        function set.eta(net,eta)
            msg = 'eta (learning rate) must be a real number greater than 0';
            if ~isnumeric(eta) || eta<=0 || ~isfinite(eta) || isempty(eta)
                error(msg);
            else
                net.eta = eta;
            end
        end
%% 
        function set.etaDecay(net,etaDecay)
            msg = 'etaDecay (learning rate decay factor) must be a real number between 0 and 1';
            if ~isnumeric(etaDecay) || etaDecay<=0 || etaDecay>1  || ~isfinite(etaDecay) || isempty(etaDecay)
                error(msg);
            else
                net.etaDecay = etaDecay;
            end
        end
%% 
        function set.batchSize(net,batchSize)
            msg = 'batchSize must be an integer';
            if ((batchSize-round(batchSize))~=0)
                error(msg);
            else
                net.batchSize = batchSize;
            end
        end
%% 
        function set.errStopThreshold(net,errStopThreshold)
            msg = 'errStopThreshold (minimum trheshold for the MSE) must be a real number greater than 0';
            if ~isnumeric(errStopThreshold) || errStopThreshold<=0 || ~isfinite(errStopThreshold) || isempty(errStopThreshold)
                error(msg);
            else
                net.errStopThreshold = errStopThreshold;
            end
        end
%% 
        function set.dissipativeTraining(net,dissipativeTraining)
            msg = 'dissipativeTraining must be a boolean (0: classic training, 1:dissipative training)';
            if ~( dissipativeTraining==0 || dissipativeTraining==1 )
                error(msg);
            else
                net.dissipativeTraining = dissipativeTraining;
            end
        end
%% 
        function set.gradientNormalization(net,gradientNormalization)
            msg = 'gradientNormalization must be a boolean';
            if ~( gradientNormalization==0 || gradientNormalization==1 )
                error(msg);
            else
                net.gradientNormalization = gradientNormalization;
            end
        end
%%
        function initialization(net,inputSize,outputSize)   
            net.inputDim = inputSize;
            net.outputDim = outputSize;
            net.W = weightsInitialization(net.HU,net.inputDim,net.weightsNormalization);
            net.O = weightsInitialization(net.outputDim,net.HU,net.weightsNormalization);
            if net.dissipativeTraining
               [net.A,net.B,net.theta,net.order,net.solutions] = makeSystemMatrix(net.parameters);
                net.M = expm(net.A*net.tau); % homogeneus matrix for dynamic system update
                net.U = net.eta*expm(net.A*net.tau/2)*net.B; % coefficient matrix for dynamic system update
                net.HN = zeros(net.order,numel(net.W));
                net.ON = zeros(net.order,numel(net.O));
                net.HN(1,:) = net.W(:)';
                net.ON(1,:) = net.O(:)';
            end
        end
        
%%
        function evaluation(net,X)
            % evaluate the output of the net the input-size-by-n_of_samples
            % matrix X
            
            net.ah = net.W*[ones(1,size(X,2));X];
            net.z = transferFunction(net.ah,net.activation);
            net.ao = net.O*[ones(1,size(X,2));net.z];
            net.o = transferFunction(net.ao,net.outputFunction);
        end
         
%%
        function learn(net,Data)
            % calculate the net gradients on structure Data: 
            % Data.X: input-size-by-n_of_samples matrix
            % Data.Y: output-size-by-n_of_samples matrix
            
            if ~isfinite(Data.Y(1))
                net.dW = zeros(net.HU,net.inputDim+1);
                net.dO = zeros(net.outputDim,net.HU+1);
            else
                %       Forward
                net.evaluation(Data.X);
                %       Error evaluation
                delta = net.o - Data.Y;
                %       Backward propagation    
                delta =  backprop(delta,net.o,net.ao,net.outputFunction);
                net.dO = delta*[ones(size(Data.X,2),1),net.z'];
                delta = net.O(:,2:end)'*delta;
                delta =  backprop(delta,net.z,net.ah,net.activation);
                net.dW = delta*[ones(size(Data.X,2),1),Data.X'];
                %       Normalization       
                if net.gradientNormalization
                    if max(max(abs(net.dW)))>0
                        net.dW = net.dW/max(max(abs(net.dW)));
                    end
                    if max(max(abs(net.dO)))>0
                        net.dO = net.dO/max(max(abs(net.dO)));
                    end
                end
            end
            
        end
        
%%
        function updating(net)
            % net updating 
            
            if net.dissipativeTraining
                    net.HN = net.M*net.HN + net.U*net.dW(:)';
                    net.ON = net.M*net.ON + net.U*net.dO(:)';
                    net.W(:) = net.HN(1,:)';
                    net.O(:) = net.ON(1,:)';
                else
                    net.W = net.W - net.eta*net.dW;
                    net.O = net.O - net.eta*net.dO;
            end           
        end
        
%%
        function train(net,Data,max_epochs)
            % train net on Data:
            % Data.X: input-size-by-n_of_samples training input matrix
            % Data.Y: output-size-by-n_of_samples matrix training targets matrix
            % Data.Xtest: input-size-by-n_of_samples test input matrix
            % Data.Ytest: output-size-by-n_of_samples matrix test targets matrix
            
            if isempty(net.W)
                net.initialization(size(Data.X,1),size(Data.Y,1));
            end
            
            if ~net.dissipativeTraining
                Data.X = Data.X(:,isfinite(Data.Y(1,:)));
                Data.Y = Data.Y(:,isfinite(Data.Y(1,:)));
            end

            if net.batchSize >= size(Data.X,2)
                lastBatchSize = size(Data.X,2);
                nUpdating = 0;
            else
                lastBatchSize = rem(size(Data.X,2),net.batchSize);
                nUpdating = floor(size(Data.X,2)/net.batchSize);
            end

            trial = 1;
            Del = repmat('\b',1,11);

            while trial<=max_epochs
                fprintf('Epochs of Training: %6i/%6i - MSE: %2.6f\n',trial,max_epochs,net.trainMSE(end));
                %	Training
                fprintf('   Training on Batch:           ');
                
                for i=1:nUpdating
                    fprintf(Del);
                    fprintf('%5i/%5i',i,nUpdating+(lastBatchSize>0));
                    Batch.X = Data.X(:,net.batchSize*(i-1)+1:net.batchSize*i);
                    Batch.Y = Data.Y(:,net.batchSize*(i-1)+1:net.batchSize*i);
                    net.learn(Batch);
                    net.updating;
                end
                if lastBatchSize>0
                    fprintf(Del);
                    fprintf('%5i/%5i',nUpdating+(lastBatchSize>0),nUpdating+(lastBatchSize>0));
                    Batch.X = Data.X(:,end-lastBatchSize+1:end);
                    Batch.Y = Data.Y(:,end-lastBatchSize+1:end);
                    net.learn(Batch);
                    net.updating;
                end
                                
                %	Performance Computing
                
                net.evaluation(Data.X(:,isfinite(sum(Data.Y,1)))); 
                [net.trainAccuracy(end+1),net.trainMSE(end+1)] = performance(net.o,Data.Y(:,isfinite(sum(Data.Y,1))));
                net.evaluation(Data.Xtest(:,isfinite(sum(Data.Ytest,1))));
                [net.testAccuracy(end+1),net.testMSE(end+1)] = performance(net.o,Data.Ytest(:,isfinite(sum(Data.Ytest,1))));

                %   Stopping criterion      
                if net.trainMSE(end)<net.errStopThreshold
                    break;
                end   
                trial = trial +1;
                net.eta = max(net.eta*net.etaDecay,1e-5);
                fprintf('\n');
            end
            fprintf('Accuracy: %1.2f \n',net.trainAccuracy(end));
        end
        
    end
end

%%



function [Accuracy,MSE] = performance(Predictions,Targets)
% performance : calculate the prediction accuracy and MSE
%
%     [Accuracy,MSE] = performance(Predictions,Targets)
%
%     Predictions: output_size-by-n_of_samples matrix of predictions
%     Targets : output_size-by-n_of_samples matrix of targets 
 

Targets = Targets(:,isfinite(sum(Targets,1)));
Predictions = Predictions(:,isfinite(sum(Targets,1)));
N = numel(Targets);

if size(Targets,1)>1
    Accuracy = mean(vec2ind(Predictions)==vec2ind(Targets));
else
    Accuracy = mean((Predictions>.5)==(Targets>.5));
end
MSE = (0.5/N)*sum(sum((Predictions - Targets).^2,1),2);
end


function delta = backprop(delta,Y,X,activation)
% backprop: propagate the error 'delta' through of a NN-layer
%
%     delta = backprop(delta,Y,X,activation)
%     delta = backprop(delta,Y,X,activation,A,S)
%
%     delta : error on the upper layer
%     Y : layer output
%     X : activation input
%     activation: string containing the name of the desired activation 
%     A,S : possible scaling parameter for 'tanh'
%
%     delta: output_size-by-n_of_samples vector of error on layer's output


A=1.7159;
S=2/3;

switch activation
    case 'softmax'
        dim = size(Y);
        delta = reshape(sum(repmat(Y(:)',dim(1),1).*(repmat(eye(dim(1)),1,dim(2))-reshape(reshape(repmat(Y,dim(1),1),[],1),dim(1),[])).*(reshape(reshape(repmat(delta,dim(1),1),[],1),dim(1),[])),1)',dim(1),[]);
    case 'logsig'
        Sigma = Y.*(1-Y);
        delta = delta.*Sigma;
    case 'tanh'
        Sigma = 4*A*S*exp(-2*S*X)./(1+exp(-2*S*X)).^2;
        delta = delta.*Sigma;
    case 'rectifier'
       Sigma = Y>0;
       delta = delta.*Sigma;        
end
end


function W = weightsInitialization(output_dim,input_dim,type)
% weights_init : random weights initialization and normalization


%random weights initialization
W = rand(output_dim,input_dim+1);

switch type   
    case'standard'
        %Subtract mean of each Column from data
        W=W-repmat(mean(W),output_dim,1);
        %normalize each observation by the standard deviation of that variable
        W=W./repmat(std(W,0,1),output_dim,1);
    case 'scaling'
        % determine the maximum value of each colunm of an array
        Max=max(W);
        % determine the minimum value of each colunm of an array
        Min=min(W);
        %array that contains the different between the maximum and minimum value for each column
        Difference=Max-Min;    
        %subtract the minimum value for each column
        W=W-repmat(Min,output_dim,1);
        %Column by the difference between the maximum and minimum value 
        W=W./repmat(Difference,output_dim,1);
    case 'global'
        W = ((W-mean(W(:)))/std(W(:)))*1e-3;
    case 'decreasing'
        scaling = 1e-3;
        W = W*scaling;
    case 'xavier'
        s = sqrt(6) / sqrt(output_dim+input_dim);
        W = W*2*s-s;
    case 'norm'
        W = (W-0.5)*0.2;
end
end


function Y = transferFunction(input,activation,A,S)
% transferFunction: compute the nonlinear activation
%
%     Y = transferFunction(input,activation)
%     Y = transferFunction(input,activation,A,S)
%
%     input : input_size-by-n_of_samples matrix layer input
%     activation: string containing the name of the desired activation 
%     A,S : possible scaling parameter for 'tanh'
%
%     Y: activation output

if nargin<3
    A=1.7159;
    S=2/3;
end

switch activation
    case 'softmax'
        Y = softmax(input);
    case 'logsig'
        Y = logsig(input);
    case 'tanh'
        Y = A*(2./(1+exp(-2*S*input))-1);
%         Y = A*tanh(S*(x));
    case 'rectifier'
       Y = max(input,0);
    case 'linear'
       Y = input;
end
end




