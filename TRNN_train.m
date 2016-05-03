function net = TRNN_train(net,TrainData,TestData,eta,tau,iterations)
% TRNN_train : save in the structure 'net' the variables after the training 
%              of a NN with the Time-Regularization algorithm;the structure
%              can be trained many times mantaining the data dimension
%
%     net = TRNN_train(net,TrainI,TrainT,TestI,TestT,eta,tau,iterations)
%
%     net: structure created by function TRNN_initialize or already trained
%          with TRNN_train(mantain data dimension)
%     TrainData: (size+number_of_classes)-by-number_of_samples matrix 
%                containing the training data and its targets NaN/Inf means 
%                unlabeled sample 
%     TestData: (size+number_of_classes)-by-number_of_samples matrix 
%                containing the test data and its targets NaN/Inf means 
%                unlabeled sample
%     eta: learning rate (1/mu)
%     tau: time sampling step of the system
%     iterations: number of epochs of training over the data
%
% Author: Alessandro Rossi (2016)
%         rossi111@unisi.it
 
tic;    

if isfield(net,'eta') % checking and saving of eta
    if net.eta ~= eta
        warning('Parameter eta has been changed');
    end
end
net.eta = eta;
if isfield(net,'tau') % checking and saving of tau       
    if net.tau ~= tau
        warning('Parameter tau has been changed');
    end
end
net.tau = tau;

net.MSE=[net.MSE zeros(1,iterations)];
net.Accuracy=[net.Accuracy zeros(1,iterations)];
net.TestMSE=[net.TestMSE zeros(1,iterations)];
net.TestAccuracy=[net.TestAccuracy zeros(1,iterations)];

% filename to save data
taustr = num2str(tau);etastr = num2str(eta);
filename=[inputname(2),'_tau',taustr(taustr~='.'),'_eta',etastr(etastr~='.'),...
               '_HU',num2str(net.Units),'_epochs',num2str(length(net.MSE))];



%     calculating the matrices for the updating formula
net.M = expm(net.A*tau);
net.U = net.eta*expm(net.A*tau/2)*net.B;
    
%     Time-width of data, check the value of e^{Theta*Tau*N}
net.DataWidth = size(TrainData,2)*net.tau ;
net.DissipationValue = exp(net.DataWidth*net.theta);

Train_lab = size(TrainData(:,isfinite(TrainData(end,:))),2);

iter = 0;

while iter < iterations 
    fprintf('Iterations left: %f \n',iterations-iter);    
    Pred_ok = 0;
    net.Epochs = net.Epochs+1;
    for k = 1:size(TrainData,2)
%             Neurons updating
        net.W = net.M*net.W+net.U*net.V_w(1:net.HiNe);
        net.C = net.M*net.C+net.U*net.V_c(1:net.OutNe);
%             Saving neurons in matrices
        net.HM(1:net.HiNe)=net.W(1:net.order:(net.HiNe*net.order));
        net.OM(1:net.OutNe)=net.C(1:net.order:(net.OutNe*net.order));
%               Evaluation
        if isfinite(TrainData(end,k)) % Calculating SE,Accuracy,Penalty's  
                                       % Gradients for labeled instances
            x = [1;TrainData(1:net.InputDim,k)];
            a = net.HM*x;
            z = max(a,0);
            y = net.OM*[ones(1,size(z,2));z];
%                 output layer error
            Delta = y-TrainData(1+net.InputDim:end,k);
%                 hidden layer error
            delta = sign(z).*(net.OM(:,2:(net.Units+1))'*Delta);        
            net.V_c = Delta * [ 1 ; z ]' ; % output matrix gradients
            net.V_w = delta * x' ; % hidden matrix gradients
            
            net.MSE(net.Epochs) = net.MSE(net.Epochs)+...
                                  (Delta'*Delta)/(length(net.OutputDim)*2);
            [~,IY]=max(y,[],1);
            [~,IT]=max(TrainData(1+net.InputDim:end,k),[],1);
            Pred_ok = Pred_ok + (IT==IY) ; 
        else % zero-settimg of Penalty's Gradients for unlabeled instances
            net.V_w = zeros(size(net.HM));
            net.V_c = zeros(size(net.OM));
        end
    end    
    iter = iter + 1;
    net.MSE(net.Epochs)=net.MSE(net.Epochs)/Train_lab;% MSE on Training Set            
    net.Accuracy(net.Epochs)=Pred_ok/Train_lab; % Accuracy evaluation on
                                                % Training Set   
    %     performance evaluation on Test Set
    [net.TestMSE(net.Epochs),net.TestAccuracy(net.Epochs)] = ...
                         PerformanceEval(net,TestData(1:net.InputDim,:),...
                                           TestData(1+net.InputDim:end,:));
                                       
    fprintf('Mean Square Error= %f \n',net.MSE(net.Epochs));
    save(filename,'net');
end

toc;



