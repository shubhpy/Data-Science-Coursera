data_test = load('test_data.csv');
Xk_test = data_test(:, [2,3,4,5,6,7,8]); Xh_test = data_test(:, [9,10,11,12,13,14,15]);
X_test=Xk_test-Xh_test;

[m, nk] = size(X_test);
X_test = [ones(m, 1) X_test];
theta=[4.239170826453695e-05;3.444963245159685e-07;2.079054654512174e-07;1.588196371510539e-07;8.607226625181769e-07;1.972062090153033e-05;5.955326072596097e-05;5.464304583160195e-05];
pt = predict(theta, X_test);