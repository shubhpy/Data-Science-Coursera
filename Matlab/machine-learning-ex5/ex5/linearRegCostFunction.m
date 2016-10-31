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

J = (1/(2*m))* sum(((X*theta)-y).^2);
%Written when X was 12x2 and theta was 2x1
J = J + sum(sum(theta(2:end,:).^2))*(lambda/(2*m));

for k=1:size(theta)
    derivative=sum((X*theta - y).*X(:,k));
    if k>1
        grad(k)=(derivative/m)+(lambda*theta(k)/m);
    else
        grad(k)=(derivative/m);
    end
end
% =========================================================================

grad = grad(:);

end
