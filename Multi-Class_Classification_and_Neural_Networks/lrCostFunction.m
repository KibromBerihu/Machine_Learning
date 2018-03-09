clcfunction [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% cost function elements 
tempOne= log(sigmoid(X*theta)); 
tempTwo= log(1-(sigmoid(X*theta))); 
temp = [ y, 1-y].*[tempOne, tempTwo]; % multiply element wise

%  Cost Function with regularization, considering that the first theta will....
% not included in the regularizatin part. 
J= (1/m)*sum(-sum(temp(:,1))-sum(temp(:,2))) +  lambda/(2*m)*sum(theta(2:end, 1).^2);


% Gradient computation
% compute grad(1,1) separately because not involved in regularization
grad(1,1)= (1/m)*sum((sigmoid(X*theta)-y).*X(:,1));
% the rest of 2:end of gradient is computed as
grad(2:end,:)= ((1/m)*((sigmoid(X*theta)-y)'*X(:,2:end)))' + (lambda/m) *(theta(2:end));
% =============================================================

grad = grad(:);
end
