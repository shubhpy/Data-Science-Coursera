function [p_vec, error_train, error_val] = ...
    validationCurve_ModelSelection(X, y, Xval, yval,lambda)

p_vec = [1 2 3 4 5 6 7 8 9 10 11 12];
error_train = zeros(length(p_vec), 1);
error_val = zeros(length(p_vec), 1);

m = size(X, 1);
mval=size(Xval,1);

for i = 1:length(p_vec)
     p = p_vec(i);
    fprintf('Running for (Degree = %f)\n\n',p);
    X_p = polyFeatures(X, p);
    [X_p, mu, sigma] = featureNormalize(X_p);
    
    Xval_p = polyFeatures(Xval, p);
    [Xval_p, mu, sigma] = featureNormalize(Xval_p);
    [THETA] = trainLinearReg(X_p, y, lambda);
    error_train(i) = (1/(2*m))* sum((X_p*THETA-y).^2);
    error_val(i) = (1/(2*mval))* sum((Xval_p*THETA-yval).^2);
end

end
