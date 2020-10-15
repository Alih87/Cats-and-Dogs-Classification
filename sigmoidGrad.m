function s = sigmoidGrad(z)
s = zeros(size(z));
s = s + sigmoid(z).*(1-sigmoid(z));
end

