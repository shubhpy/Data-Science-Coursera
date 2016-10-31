x = load('ex5Logx.dat'); 
y = load('ex5Logy.dat');
figure

% Find the indices for the 2 classes
pos = find(y); neg = find(y == 0);

plot(x(pos, 1), x(pos, 2), '+')
hold on
plot(x(neg, 1), x(neg, 2), 'o')

%x = map_feature(u, v)

% Define the ranges of the grid
u = linspace(-1, 1.5, 200);
v = linspace(-1, 1.5, 200);

% Initialize space for the values to be plotted
z = zeros(length(u), length(v));

% Evaluate z = theta*x over the grid
for i = 1:length(u)
    for j = 1:length(v)
        % Notice the order of j, i here!
        z(j,i) = map_feature(u(i), v(j))*theta;
    end
end

% Because of the way that contour plotting works
% in Matlab, we need to transpose z, or
% else the axis orientation will be flipped!
z = z'
% Plot z = 0 by specifying the range [0, 0]
contour(u,v,z, [0, 0], 'LineWidth', 2)