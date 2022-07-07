function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y);
h = sigmoid(X*theta);

temp = theta;
temp(1) = 0;

J = -(y' * log(h) + (1-y)' * log(1-h) -lambda/2 * temp' * temp)/m;

grad = (X' * (h-y) + lambda*temp)/m;
end
