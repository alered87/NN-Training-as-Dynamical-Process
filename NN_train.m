function net = NN_train(net,TrainData,TestData,eta,iterations,batch_size)
% NN_train : save in the structure 'net' the variables after the standard  
%            training of a NN initialized with TRNN_initialize
%
%     net = NN_train(net,TrainData,TestData,eta,iterations,batch_size)
%
%     net: structure created by function TRNN_initialize or already trained
%          with NN_train (mantain data dimension)
%     TrainData: (size+number_of_classes)-by-number_of_samples matrix 
%                containing the training data and its targets NaN/Inf means 
%                unlabeled sample 
%     TestData: (size+number_of_classes)-by-number_of_samples matrix 
%                containing the test data and its targets NaN/Inf means 
%                unlabeled sample
%     eta: learning rate
%     iterations: number of epochs of training over the data
%     batch_size: number of samples to process before weights updating
%
% Author: Alessandro Rossi (2016)
%         rossi111@unisi.it

 
tic;    

% removing possible unlabeled samples from Test Data
TrainData=TrainData(:,isfinite(TrainData(end,:)));
TestData=TestData(:,isfinite(TestData(end,:)));

net.MSE=[net.MSE zeros(1,iterations)];
net.Accuracy=[net.Accuracy zeros(1,iterations)];
net.TestMSE=[net.TestMSE zeros(1,iterations)];
net.TestAccuracy=[net.TestAccuracy zeros(1,iterations)];

net.eta = eta;

n_last_batch = rem(size(TrainData,2),batch_size);
n_updating = floor(size(TrainData,2)/batch_size);
if batch_size == size(TrainData,2)
    n_last_batch = size(TrainData,2);
end

% filename to save data
etastr = num2str(eta);

filename=['NN_',inputname(2),'_bs',num2str(batch_size),...
          '_eta',etastr(etastr~='.'),...
          '_HU',num2str(net.Units),'_epochs',num2str(length(net.MSE))];
iter = 1;

while iter <= iterations
    fprintf('Iteration: %i/%i \n',iter,iterations);    
    net.Epochs = net.Epochs+1;
    fprintf('   Training on Batch:             ');
%         training on the batch
    for i = 1:n_updating
        fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b%6i/%6i',i,n_updating+(n_last_batch>0));
        net.V_w = zeros(size(net.HM));
        net.V_c = zeros(size(net.OM));
        net = back_prop(net,TrainData(:,(i-1)*batch_size+1:i*batch_size));
        net.V_w = net.V_w / max(max(abs(net.V_w)));
        net.V_c = net.V_c / max(max(abs(net.V_c)));
        net.HM = net.HM - net.eta*net.V_w;
        net.OM = net.OM - net.eta*net.V_c;
    end
    if n_last_batch>0
        fprintf('\b\b\b\b\b\b\b\b\b\b\b\b\b%6i/%6i',n_updating+(n_last_batch>0),n_updating+(n_last_batch>0));
        net.V_w = zeros(size(net.HM));
        net.V_c = zeros(size(net.OM));
        net = back_prop(net,TrainData(:,end-n_last_batch+1:end));
        net.V_w = net.V_w / max(max(abs(net.V_w)));
        net.V_c = net.V_c / max(max(abs(net.V_c)));
        net.HM = net.HM - net.eta*net.V_w;
        net.OM = net.OM - net.eta*net.V_c;
    end
    fprintf('\n');
    iter = iter + 1;
        %     performance evaluation on Training Set
    [net.MSE(net.Epochs),net.Accuracy(net.Epochs)] = ...
                        PerformanceEval(net,TrainData(1:net.InputDim,:),...
                                          TrainData(1+net.InputDim:end,:));
    %     performance evaluation on Test Set
    [net.TestMSE(net.Epochs),net.TestAccuracy(net.Epochs)] = ...
                         PerformanceEval(net,TestData(1:net.InputDim,:),...
                                           TestData(1+net.InputDim:end,:));                                       
                                       
    fprintf('Mean Square Error= %f \n',net.MSE(net.Epochs));
    save(filename,'net');
end

toc;
end


function net = back_prop(net,Data)

for k = 1:size(Data,2)
    x = [1;Data(1:net.InputDim,k)];
    a = net.HM*x;
    z = max(a,0);
    y = net.OM*[ones(1,size(z,2));z];
%       output layer error
    Delta = y-Data(1+net.InputDim:end,k);
%       hidden layer error
    delta = sign(z).*(net.OM(:,2:(net.Units+1))'*Delta);        
    net.V_c = net.V_c + Delta * [ 1 ; z ]' ;
    net.V_w = net.V_w + delta * x' ;
end
end


