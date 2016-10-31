function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
% theta = inv(X'*X)*(X'*y); 
% using multivariate approach  -3.636063 1.166989
% using univariate approach    -3.630291 1.166362
% using normal equation        -3.8958   1.1930

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
for i = 1:m
    J = J + (theta(1)+(theta(2)*X(i,2))-y(i)).^2;
end
J = J/(2*m);

% =========================================================================

end
