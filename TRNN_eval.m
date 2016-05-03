function Prediction = TRNN_eval(net,input) 
% TRNN_eval : evaluate the net on a single input instance and calulate the
%             gradient of the penalty function w.r.t. the weights(neurons)
%
%     Prediction = TRNN_eval(net,input)
%
%     net: existing structure containing the variables of the model
%     input: size-by-n_of_samples input matrix
%     
%     Prediction: evaluation of the NN on 'input'
%
% Author: Alessandro Rossi (2016)
%         rossi111@unisi.it

x = [ones(1,size(input,2));input];
a = net.HM*x;
z = reshape(max(a(:),0),net.Units,size(input,2));
Prediction = net.OM*[ones(1,size(z,2));z];  

end