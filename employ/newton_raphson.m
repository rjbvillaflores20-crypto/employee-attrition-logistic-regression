function beta = newton_raphson(X, y)
% Manual logistic regression using Newton-Raphson

[n, k] = size(X);
beta = zeros(k,1);

tol = 1e-6;
max_iter = 100;

for iter = 1:max_iter
    
    % Predicted probabilities
    p = 1 ./ (1 + exp(-X * beta));
    
    % Weight matrix
    W = diag(p .* (1 - p));
    
    % Gradient
    grad = X' * (y - p);
    
    % Hessian
    H = -X' * W * X;
    
    % Update rule
    beta_new = beta - H \ grad;
    
    % Convergence check
    if norm(beta_new - beta) < tol
        fprintf('Converged in %d iterations\n', iter);
        break;
    end
    
    beta = beta_new;
end

end
