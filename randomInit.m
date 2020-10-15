function randTheta = randomInit(inputL, hiddenL)

epsilon = 0.01;
randTheta = zeros(hiddenL, inputL + 1);
randTheta = randTheta + 2*epsilon*rand(hiddenL, inputL + 1) - epsilon;
randTheta = randTheta(:);

end
