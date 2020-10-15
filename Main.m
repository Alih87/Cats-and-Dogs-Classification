Data = load("trainingData.mat");
x = Data.data(:, 1:10000)./255;
x_train = x(1:6400, :);
y_train = Data.data(1:6400, 10001);

%sel = randperm(size(x_train, 1));
%sel = sel(1:100);

%displayData(x_train(sel, :));

%fprintf('Program paused. Press enter to continue.\n');
%pause;

inputL = 10000;
hiddenL = 3000;
hiddenL1 = 25;
numL = 2;
lambda = 0.001;
initheta1 = randomInit(inputL, hiddenL);
initheta2 = randomInit(hiddenL, hiddenL1);
initheta3 = randomInit(hiddenL1, numL);
initheta = [initheta1(:); initheta2(:); initheta3(:)];

% Cost for randomly initialized weights

J = cost(x_train, y_train, inputL, hiddenL, hiddenL1, numL, lambda, initheta);
fprintf("\nCost for randomly initialized weights: %f\n", J);

options = optimset("MaxIter", 200, "GradObj", "on");
costFunc = @(p) cost(x_train, y_train, inputL, hiddenL, hiddenL1, numL, lambda, p);

[Theta, JJ] = fmincg(costFunc, initheta, options);
fprintf("Cost for learned weights: %f", JJ);

Theta = Theta(:);

save learnedTheta.mat Theta;

theta1 = reshape(Theta(1:hiddenL*(inputL + 1)), hiddenL , inputL + 1);
theta2 = reshape(Theta(1 + (hiddenL*(inputL + 1)):(hiddenL1*(hiddenL + 1))), hiddenL1, hiddenL + 1);
theta3 = reshape(Theta(1 + (hiddenL1*(hiddenL + 1)):end), numL, hiddenL1 + 1);

displayData(theta1(:, 2:end));
pause;

pre = predict(theta1, theta2, x_train);
p = mean(double(pre == y_train))*100;

fprintf("\n\nThe accuracy of the algorithm is: %f", p);