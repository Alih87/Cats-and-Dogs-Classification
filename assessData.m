Data = load("trainingData.mat");
x = Data.data(:,1:900)./255;
x_train = x(1:4800, :);
y_train = Data.data(1:4800, 901);
x_cv = x(4801:end, :);
y_cv = Data.data(4801:end, 901);

inputL = 900;
hiddenL = 25;
numL = 2;
lambda = 0.01;


[J_train, J_cv, m_train, m_cv] = Diagnose(x_train, y_train, x_cv, y_cv, lambda);

hold on;
plot(m_train, J_train, "b-");
plot(m_cv, J_cv, "r-");
xlabel("m")
ylabel("J")
legend("Training Data","Cross Validation")
