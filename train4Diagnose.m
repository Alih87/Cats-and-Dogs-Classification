function Theta = train4Diagnose(x, y, lambda)
inputL = 900;
hiddenL = 25;
numL = 2;
initheta1 = randomInit(inputL, hiddenL);
initheta2 = randomInit(hiddenL, numL);
initheta = [initheta1; initheta2];

options = optimset("MaxIter", 200, "GradObj", "on");
costFunc = @(p) cost(x, y, inputL, hiddenL, numL, lambda, p);

Theta = fmincg(costFunc, initheta, options);
end

