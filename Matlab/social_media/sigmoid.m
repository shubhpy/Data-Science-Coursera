function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   J = SIGMOID(z) computes the sigmoid of z.


% You need to return the following variables correctly 
%z=[1,2,3,4,5];
%z=25;
[m,n]=size(z);
g = zeros(m,n);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

g =1.0 ./(1.0+exp(-z));
% =============================================================

end
