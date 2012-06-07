function [J, grad] = costFunction(theta, X, y)
    %COSTFUNCTION Compute cost and gradient for logistic regression
    %   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    %   parameter for logistic regression and the gradient of the cost
    %   w.r.t. to the parameters.

    m = length(y); % number of training examples
    o = sigmoid(X*theta);
    e = o-y;

    J = sum((-y.*log(o)) - (1-y).*log(1-o))/m;
    grad = sum(repmat(e,1,size(X,2)).*X)'/m;
end