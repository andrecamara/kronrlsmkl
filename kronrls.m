function [A] = kronrls(k1,k2,y, lambda)
    if ~exist('lambda','var') || isempty(lambda)
        lambda = 1;
    end
    [Qa,la] = eig(k1);
    [Qb,lb] = eig(k2);
    l = kron(diag(lb)',diag(la));
    inverse = l ./ (l + lambda);
    m1 = Qa' * y * Qb;
    m2 = m1 .* inverse;
    A = Qa * m2 * Qb';    
end