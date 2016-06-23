function [best_lambda, bestaupr] = optimize_lambda(K1, K2, y)
    exponents   = [-15:5:30];
    bestaupr    = 0;
    best_lambda = 1;
    
%     nruns   = 1;
    nfolds  = 5;
%     cv_type = 'pairs';
    
%     crossval_idx = crossvalind('Kfold', length(y(:)), nfolds);
    crossval_idx = crossvalind('Kfold',y(:),5);
    
    for e = exponents
        predictions = zeros(size(y));

        for fold=1:nfolds    
            test_idx  = find(crossval_idx==fold);
            y_train = y;
            y_train(test_idx) = 0;
            
%             fprintf('positive=%d, ', sum(y_train(:)));
            
%             if sum(y_train(:)) <= 0
%                 sum(y_train(:))
%             end
            
            lambda = 2^e;
            
            [ y2 ] = kronrls( K1, K2, y_train, lambda);
            predictions(test_idx) = y2(test_idx); 
        end
        yy = y;
        yy(yy==0)=-1;
        stats = evaluate_performance(predictions(:), yy(:), 'classification');
        
        fprintf(' (kronrls): INNER-CV, lambda=2^%d - AUPR=%.4f\n', e, stats.aupr);
        
        if stats.aupr > bestaupr
            best_lambda = lambda;
            bestaupr = stats.aupr;
        end
    end  
end