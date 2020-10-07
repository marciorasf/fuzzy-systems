% To model this system, it was used exercise_2 functions and parameters,
% once sine and cosine are the same function moved by -pi/2

% read fuzzy inference system
fis = readfis("exercise_3.fis");

% calculate samples 
x = [0 : 0.01 : 2*pi];
y = sin(x);
yApprox = transpose(evalfis(fis, x));

% calculate mean squared error
error = y-yApprox;
squaredError = error .^ 2;
mse = mean(squaredError);
disp("mean squared error = ");
disp(mse);

% plot fuzzy approximation
plot(x, y, x, yApprox);
title("Fuzzy Approximation");
xlabel('Input');
ylabel('Output') ;
legend({'y = sin(x)','y = Fuzzy(x)'},'Location','southwest');