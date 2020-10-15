function J = checkCost(x, y, inputL, hiddenL, numL, lambda, initheta)

% Initializing
J = 0;
theta1 = reshape(initheta(1:hiddenL*(inputL + 1)), hiddenL , inputL + 1);
theta2 = reshape(initheta(1 + (hiddenL*(inputL + 1)):end), numL, hiddenL + 1);
regtheta1 = theta1;
regtheta2 = theta2;
regtheta1 = regtheta1(:, 2:end);
regtheta2 = regtheta2(:, 2:end);
m = size(x,1);

% Forward Propagation
a1 = [ones(m,1), x];
a2 = sigmoid(a1*theta1');
a2 = [ones(m,1), a2];
a3 = sigmoid(a2*theta2');
Y = repmat([2,3], m, 1) == repmat(y, 1, numL);

% cost
J = J + ((-1/m)*sum(sum(Y.*log(a3) + (1-Y).*log(1-a3)))) ...
+ (lambda/(2*m)*sum((sum(sum(regtheta1.^2))) + sum(sum(regtheta2.^2))));


end
