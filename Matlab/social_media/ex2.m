%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the logistic
%  regression exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear; close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

data = load('train_data.csv');
Xk = data(:, [3,4,5,6,7,8,9]); Xh = data(:, [10,11,12,13,14,15,16]); y = data(:, 2);

%% ============ Part 2: Compute Cost and Gradient ============
%  In this part of the exercise, you will implement the cost and gradient
%  for logistic regression. You neeed to complete the code in 
%  costFunction.m

%  Setup the data matrix appropriately, and add ones for the intercept term
[mk, nk] = size(Xk);
[mh, nh] = size(Xh);

% Add intercept term to x and X_test
Xk = [ones(mk, 1) Xk];
Xh = [ones(mh, 1) Xh];

% Initialize fitting parameters
initial_theta_ = zeros(nk + 1, 1);
%initial_theta_h = zeros(nh + 1, 1);

% Compute and display initial cost and gradient
[cost_, grad_] = costFunction(initial_theta_, Xk-Xh, y);
%[cost_h, grad_h] = costFunction(initial_theta_h, Xh, y);

fprintf('Cost at initial theta (zeros) for k : %f\n', cost_);
fprintf('Gradient at initial theta (zeros) for k : \n');
fprintf(' %f \n', grad_);

%fprintf('Cost at initial theta (zeros) for h : %f\n', cost_h);
%fprintf('Gradient at initial theta (zeros) for h : \n');
%fprintf(' %f \n', grad_h);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ============= Part 3: Optimizing using fminunc  =============
%  In this exercise, you will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Set options for fminunc

options = optimset('GradObj', 'on', 'MaxIter', 300);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta_,cost_] = fminunc(@(t)(costFunction(t, Xk-Xh, y)), initial_theta_, options);

% Print theta_k to screen
fprintf('Cost at theta found by fminunc for k : %f\n', cost_);
fprintf('theta_: \n');
fprintf(' %f \n', theta_);


%[theta_h, cost_h] = ...
%	fminunc(@(t)(costFunction(t, Xh, y)), initial_theta_h, options);

% Print theta_h to screen
%fprintf('Cost at theta found by fminunc for h : %f\n', cost_h);
%fprintf('theta_h: \n');
%fprintf(' %f \n', theta_h);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============== Part 4: Predict and Accuracies ==============
%  After learning the parameters, you'll like to use it to predict the outcomes
%  on unseen data. In this part, you will use the logistic regression model
%  to predict the probability that a student with score 45 on exam 1 and 
%  score 85 on exam 2 will be admitted.
%
%  Furthermore, you will compute the training and test set accuracies of 
%  our model.
%
%  Your task is to complete the code in predict.m

%  Predict probability for a student with score 45 on exam 1 
%  and score 85 on exam 2 

% Compute accuracy on our training set
p = predict(theta_, Xk-Xh);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;