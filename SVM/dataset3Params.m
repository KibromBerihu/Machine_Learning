function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% 
% given examples of C and sigma
C_vec = [30; 0.1; 0.3; 1.3; 10; 30; 0.09]; 
sigma_vec = C_vec; 
prediction =[]; 
for c=1:length(C_vec)
    for s=1:length(sigma_vec)
        model = svmTrain(X,y, C_vec(c), @(x1,x2) gaussianKernel(x1, x2, sigma_vec(s))); 
        predictions =svmPredict(model, Xval); 
        [indx, ~] = find(predictions == yval); % find when they are eqqual values index
        prediction = [prediction; double((numel(indx)/numel(yval))) C_vec(c) sigma_vec(s)];  % count the number of equal value index
    end
end 
[~,I] = max(prediction(:,1));
C =prediction(I, 2);  % C_vec value 
sigma = prediction(I, 3); % sigma_vec value 

% =========================================================================

end
