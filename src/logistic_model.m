function p = logistic_model(X, beta)
% Logistic (sigmoid) function
p = 1 ./ (1 + exp(-X * beta));
end
