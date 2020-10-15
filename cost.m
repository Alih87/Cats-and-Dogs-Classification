function [J, grad] = cost(x, y, inputL, hiddenL, hiddenL1, numL, lambda, initheta)

% Initializing
J = 0;
L1 = hiddenL*(inputL + 1);
L2 = (hiddenL1*(hiddenL + 1)) + L1;
L3 = (numL*(hiddenL1 + 1)) + L2;

theta1 = reshape(initheta(1:L1), hiddenL, (inputL + 1));
theta2 = reshape(initheta((L1+1):L2), hiddenL1, (hiddenL+1));
theta3 = reshape(initheta((L2+1):L3), numL, (hiddenL1 + 1));

regtheta1 = theta1;
regtheta2 = theta2;
regtheta3 = theta3;
regtheta1 = regtheta1(:, 2:end);
regtheta2 = regtheta2(:, 2:end);
regtheta3 = regtheta3(:, 2:end);
Theta1_grad = zeros(size(theta1));
Theta2_grad = zeros(size(theta2));
Theta3_grad = zeros(size(theta3));
del1 = zeros(size(theta1));
del2 = zeros(size(theta2));
del3 = zeros(size(theta3));
m = size(x,1);

% Forward Propagation
a1 = [ones(m,1), x];
a2 = sigmoid(a1*theta1');
a2 = [ones(m,1), a2];
a3 = sigmoid(a2*theta2');
a3 = [ones(m,1), a3];
a4 = sigmoid(a3*theta3');
Y = repmat([2,3], m, 1) == repmat(y, 1, numL);

% cost
J = J + ((-1/m)*sum(sum(Y.*log(a4) + (1-Y).*log(1-a4)))) ...
+ (lambda/(2*m)*sum((sum(sum(regtheta1.^2))) + sum(sum(regtheta2.^2)) + sum(sum(regtheta3.^2))));

% Back Propagation
for t = 1:m
    a1t = a1(t,:);
    a2t = a2(t,:);
    a3t = a3(t,:);
    a4t = a4(t,:);
    yt = Y(t,:);
    
    d4 = a4t - yt;
    d3 = (theta3'*d4').*sigmoidGrad([1; theta2*a2t']);
    d2 = (theta2'*d3').*sigmoidGrad([1; theta1*a1t']);
    del1 = del1 + d2(2:end)*a1t;
	del2 = del2 + d3(2:end)*a2t;
    del3 = del3 + d4' * a3t;
end

Theta1_grad = Theta1_grad + 1/m * del1 + (lambda/m)*[zeros(size(theta1, 1), 1), regtheta1];
Theta2_grad = Theta2_grad + 1/m * del2 + (lambda/m)*[zeros(size(theta2, 1), 1), regtheta2];
Theta3_grad = Theta3_grad + 1/m * del3 + (lambda/m)*[zeros(size(theta3, 1), 1), regtheta3];

grad = [Theta1_grad(:); Theta2_grad(:); Theta3_grad(:)];
end
