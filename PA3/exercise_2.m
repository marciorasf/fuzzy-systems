clear();
clc();

% Define train data
x = transpose((0:0.05:2*pi));
y = sin(x);
trainingData = [x y];

initFis = readfis("exercise_2.fis");

% Define anfis training options
opt = anfisOptions;
opt.InitialFIS = initFis;
opt.EpochNumber = 200;
opt.DisplayErrorValues = false;
opt.DisplayStepSize = false;
opt.DisplayANFISInformation = false;
opt.DisplayFinalResults = false;

% Train anfis
fis = anfis(trainingData, opt);

% Calculate output using trained anfis
approxY = evalfis(fis, x);

% Calculate mean squared error
error = y - approxY;
squaredError = error .^ 2;
meanSquaredError = mean(squaredError);
disp("Mean Squared Error = " + meanSquaredError);

plot(x, y, '*r', x, approxY, '.b');
title("ANFIS Approximation");
xlabel('Input'); ylabel('Output') ;
legend({'y = sin(x)','y = ANFIS(x)'},'Location','southwest');
