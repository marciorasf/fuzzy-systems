% read fuzzy inference system
fis = readfis("exercise_2.fis");

% calculate samples 
x = [-pi/2 : 0.01 : 3/2*pi];
y = cos(x);
yApprox = transpose(evalfis(fis, x));

% calculate mean squared error
error = y-yApprox;
squaredError = error .^ 2;
mse = mean(squaredError);

% plot fuzzy approximation
plot(x, y, x, yApprox);
title("Fuzzy Approximation");
xlabel('Input') 
ylabel('Output') 
legend({'y = cos(x)','y = Fuzzy(x)'},'Location','southwest')