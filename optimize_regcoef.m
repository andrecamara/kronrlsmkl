function [best_regcoef] = optimize_regcoef(K1, K2, y)
    regcoef_values      = [0:0.25:1];
    
    best_regcoef = 0;
    bestaupr=0;
    
    lambda  = 1;
    nfolds  = 5;
%     nruns   = 1;
%     cv_type = 'pairs';
    
%     crossval_idx = crossvalind('Kfold', length(y(:)), nfolds);
    crossval_idx = crossvalind('Kfold',y(:),5);
    
    for r = regcoef_values
        predictions = zeros(size(y));

        for fold=1:nfolds    
%             train_idx = find(crossval_idx~=fold);
            test_idx  = find(crossval_idx==fold);
            y_train = y;
            y_train(test_idx) = 0;
            
            regcoef = r;
            isinner=1;
            
            [ P, ~, ~ ] = kronrls_mkl( K1, K2, y_train, lambda, regcoef, '', isinner );
            predictions(test_idx) = P(test_idx); 
            
%             true_y = y;
%             true_y(true_y==0)=-1;
            
%             fprintf('\t\tpositive=%d\n', sum(y_train(:)))
            
%             tmpstats = evaluate_performance(y2(test_idx), true_y(test_idx), 'classification');
%             fprintf('\t\t INNER-CV (regcoef), fold=%d, regcoef=%.2f - AUPR=%.4f\n', fold, regcoef, tmpstats.aupr);
        end
        yy = y;
        yy(yy==0)=-1;
        stats = evaluate_performance(predictions(:), yy(:), 'classification');
        
        fprintf(' (kronrls): INNER-CV, regcoef=%.2f - AUPR=%.4f\n', regcoef, stats.aupr);
        
        if stats.aupr > bestaupr
            best_regcoef = regcoef;
            bestaupr = stats.aupr;
        end
    end  
end