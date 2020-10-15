function [J_train, J_cv, m_train, m_cv] = Diagnose(x_train, y_train, x_cv, y_cv, lambda)
q = 1;
J_train = zeros(size(x_train, 1), 1);
m_train = zeros(size(J_train, 1));
J_cv = zeros(size(x_cv, 1), 1);
m_cv = zeros(size(J_cv, 1));
inputL = 900;
hiddenL = 25;
numL = 2;

for i = linspace(1, length(x_train), 30)
    Theta = train4Diagnose(x_train(1:i,:), y_train(1:i,:), lambda);
    J_train(q) = checkCost(x_train(1:i,:), y_train(1:i,:), inputL, hiddenL, numL, 0, Theta);
    m_train(q) = i;
    J_cv(q) = checkCost(x_cv, y_cv, inputL, hiddenL, numL, 0, Theta);
    m_cv(q) = i;
    q = q + 1;
end

end
