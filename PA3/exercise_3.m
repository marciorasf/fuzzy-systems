clear();
clc();

% -------------- GET AND PROCESS DATA -------------

% Read data
rawData = load("mg.mat");
originalX = rawData.x;
nOriginalX = length(originalX);

% Format data to use on ANFIS
leftPadding = 24;
data = zeros(nOriginalX - leftPadding, 5);
for time = leftPadding + 1 : nOriginalX
    data(time - leftPadding, :) = [originalX(time-24) originalX(time-18) originalX(time-12) originalX(time-6) originalX(time)];
end

nData = length(data);
time = (leftPadding+1 : nOriginalX);

% Separate train and test data
trainRatio = 0.8;
trainLastIndex = round(trainRatio * nData);

trainTime = time(1 : trainLastIndex);
trainInput = data(1 : trainLastIndex, 1:4);
trainOutput = data(1: trainLastIndex, end);
trainData = [trainInput trainOutput];

testTime = time(trainLastIndex + 1 : end);
testInput = data(trainLastIndex + 1: end, 1:4);
testOutput = data(trainLastIndex + 1: end, end);
testData = [testInput testOutput];

% -------------- DEFINE INITIAL FIS -------------

% Distribute the membership functions evenly on the input space
genfisOpt = genfisOptions("GridPartition");

% Each input has 2 membership functions
genfisOpt.NumMembershipFunctions = [2 2 2 2];

% Use generalized bell for each input
genfisOpt.InputMembershipFunctionType = ["gbellmf" "gbellmf" "gbellmf" "gbellmf"];

% Generate inital fis
initFis = genfis(trainInput, trainOutput, genfisOpt);

% Plot the inputs' membership functions of the initial fis
% figure;
% subplot(2, 2, 1);
% plotmf(initFis, 'input', 1);
% xlabel('x(t-18)');
% subplot(2, 2, 2);
% plotmf(initFis, 'input', 2);
% xlabel('x(t-12)');
% subplot(2 , 2, 3);
% plotmf(initFis, 'input', 3);
% xlabel('x(t-6)');
% subplot(2, 2, 4);
% plotmf(initFis, 'input', 4);
% xlabel('x(t)');

% ------------------ TRAIN ANFIS ------------------

% Define anfis options
anfisOpt = anfisOptions;
anfisOpt.InitialFIS = initFis;
anfisOpt.EpochNumber = 20;
% opt.DisplayANFISInformation = true;
anfisOpt.DisplayErrorValues = false;
anfisOpt.DisplayStepSize = false;
anfisOpt.DisplayFinalResults = false;

% Train ANFIS
anfis = anfis(trainData, anfisOpt);

% ------------------ GENERATE RESULTS ------------------

% Evaluate inputs using trained ANFIS
trainAnfisOutput = evalfis(anfis, trainInput);
testAnfisOutput = evalfis(anfis, testInput);

% Calculate mean squared error
trainMeanSquaredError = mean((trainOutput - trainAnfisOutput).^2);
testMeanSquaredError = mean((testOutput - testAnfisOutput).^2);

% ------------------ DISPLAY RESULTS ------------------

% Display mean squared errors
disp("Mean Squared Error for train data = " + trainMeanSquaredError);
disp("Mean Squared Error for test data = " + testMeanSquaredError);

% Plot output comparisons
figure;
plot(trainTime, trainOutput, '*r', trainTime, trainAnfisOutput, '.b');
title("ANFIS Approximation for Train Data");
xlabel('t');
ylabel('Output') ;
legend({'Data','ANFIS(t)'},'Location','northeast');

figure;
plot(testTime, testOutput, '*r', testTime, testAnfisOutput, '.b');
title("ANFIS Approximation for Test Data");
xlabel('t');
ylabel('Output') ;
legend({'Data','ANFIS(t+6)'},'Location','northeast');