function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% hypothesis
h= X*theta;

% Regularized linear regression Cost function
J = (1/(2*m))*sum(((h-y).^2)); % with out the regularization
J = J + (lambda/(2*m))*(sum(theta.^2)-theta(1,1)^2); %  with regularazation and making zero the first theta

% Regularized linear regression gradient 
grad = X'*(h-y)/m; 
Theta_0 = theta; 
Theta_0(1,1)=0; 
grad = grad + (lambda*Theta_0)/m; % the theta(2:end) only 
% =========================================================================

grad = grad(:);

end
