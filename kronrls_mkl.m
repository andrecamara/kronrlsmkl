% function [ y2, alpha, beta ] = kronrls_mkl( K1, K2, y, lambda, regcoef, maxiter, isinner)
function [ A, alpha, beta ] = kronrls_mkl( K1, K2, y, lambda, regcoef, maxiter, isinner)
%KRONRLS_MKL Summary of this function goes here
%   INPUTS
%    'K1' : 3 dimensional array of kernels
%    'K2' : 3 dimensional array of kernels
%    'y'  : adjacency matrix
%    'lambda' : lambda regulaziation coefficient of KronRLS algorithm
%    'regcoef': regularization parameter of kernel weights
%    'maxiter': maximum number of iterations (default=20)
%
%   OUTPUTS
%    'y2': predictions
%    'alpha': weights of kernels K1
%    'beta':  weights of kernels K2

    %restrict to MKL on both K1 and K2
    assert(size(K1,3)>1 && size(K2,3)>1,'K1 and K2 must order 3 tensors');
    
    if ~exist('maxiter','var') || isempty(maxiter)
        maxiter = 20;
    end
    if ~exist('regcoef','var') || isempty(regcoef)
        regcoef = 0.2;
    end
    if ~exist('isinner','var') || isempty(isinner)
        isinner = 0;
    end
    
    nA = size(K1,3); 
    nB = size(K2,3); 
    
    %initialization of kernel weights (uniform)
    alpha(1:nA) = 1/nA;
    beta(1:nB)  = 1/nB;
  
    iter = 0;
    incr = 1000;
    limit_incr = 0.0100;
    combFval = zeros(1,maxiter);
    
    if ~isinner
        regcoef = optimize_regcoef(K1, K2, y);
    end
    
    %% iterative steps: optimize weights
    while(iter < maxiter && abs(incr) > limit_incr) 
        
        iter = iter+1;
        
        %% Step 1
        K1_comb = combine_kernels(alpha, K1);
        K2_comb = combine_kernels(beta, K2);
        
        [A] = kronrls(K1_comb, K2_comb, y, lambda);
        
        u = (y - lambda*A/2); 
        
        %% Step 2
        Ma = zeros([size(y') nA]); 
        for i=1:nA
            Ma(:,:,i) = K2_comb'*A'*(K1(:,:,i)); 
        end
        
        falpha = @(alpha)obj_function(alpha,u,Ma,lambda,regcoef);
        
        % Optimal Alpha
        [x_alpha, fval_alpha] = optimize_weights(alpha, falpha);
        
        %% Step 3
        Mb = zeros([size(y') nB]); 
        for i=1:nB
            Mb(:,:,i) = K2(:,:,i)'*A'*K1_comb; 
        end
        fbeta = @(beta)obj_function(beta,u,Mb,lambda,regcoef);
        
        % Optimal Beta
        [x_beta, fval_beta] = optimize_weights(beta, fbeta);
        
        %% update values for next iteration
        alpha = x_alpha;
        beta  = x_beta;
        
        combFval(iter) = (fval_alpha + fval_beta);
        if iter > 1
            incr =  combFval(iter-1) - combFval(iter);
        end  
        
        if ~isinner
            fprintf('ITER=%d \tlambda=%.2f \tregcoef=%.1f - (fval_alpha=%.4f, fval_beta=%.4f) \n',iter, lambda, regcoef, fval_alpha, fval_beta)
        end
    end
    
    %% Perform predictions
    K1_comb = combine_kernels(alpha, K1);
    K2_comb = combine_kernels(beta, K2);
    
    %build final model with best weights/lambda
    if ~isinner
        lambda  = optimize_lambda(K1_comb, K2_comb, y);
    end
    [A] = kronrls(K1_comb, K2_comb, y, lambda);
    
    y2 = K2_comb' * A' * K1_comb;
    y2 = y2';
end

function [x, fval] = optimize_weights(x0, fun)
    n = length(x0);
    Aineq   = [];
    bineq   = [];
    Aeq     = ones(1,n);
    beq     = 1;
    LB      = zeros(1,n);
    UB      = ones(1,n);

    options = optimoptions('fmincon','Algorithm','interior-point', 'Display', 'notify');
    [x,fval] = fmincon(fun,x0,Aineq,bineq,Aeq,beq,LB,UB,[],options);
end

function [J] = obj_function(w,u,Ma,lambda,regcoef)
    V = combine_kernels(w, Ma);
    J = norm(u-V','fro')/(2*lambda*numel(V)) + regcoef*(norm(w,2))^2;
end
