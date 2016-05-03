function [MSE,Accuracy] = PerformanceEval(net,input,labels)
% PerformanceEval : calculating the performance of the NN with variables
%                   saved in the structure 'net' 
%
%     [MSE,Accuracy] = PerformanceEval(net,input,labels)
%
%     net: structure containing the variables of a NN traine with the
%          Time-Regularization algorithm
%     input: size-by-number_of_samples matrix containg the training data
%          (without targets)
%     labels : number_of_classes-by-number_of_samples matrix containing 
%              binary targets for input(Inf/NaN means unlabeled)
%
% Author: Alessandro Rossi (2016)
%         rossi111@unisi.it

input = input(:,isfinite(labels(1,:)));
labels = labels(:,isfinite(labels(1,:)));

Prediction = TRNN_eval(net,input);

MSE = (0.5/size(input,1))*mean(sum((Prediction-labels).^2)); 

[~,IP]=max(Prediction,[],1);
[~,IY]=max(labels,[],1);
Accuracy = mean(IP==IY);

end