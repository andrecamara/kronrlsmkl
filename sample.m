clear
seed = 12345678;
rand('seed', seed);
nfolds = 5; nruns=5;

dataname = 'nr';
% dataname = 'gpcr';
% dataname = 'ic';
% dataname = 'e';


% load adjacency matrix
[y,l1,l2] = loadtabfile(['data/interactions/' dataname '_admat_dgc.txt']);
run_aupr  = [];
alpha_m = [];

for run=1:nruns
    % split folds
%     crossval_idx = crossvalind('Kfold', length(y(:)), nfolds);
    crossval_idx = crossvalind('Kfold',y(:),5);
    fold_aupr = [];

    for fold=1:nfolds
        fprintf('---------------\nRUN %d - FOLD %d \n', run, fold)
        train_idx = find(crossval_idx~=fold);
        test_idx  = find(crossval_idx==fold);

        y_train = y;
        y_train(test_idx) = 0;

        %% load kernels
        k1_paths = {['data/kernels/' dataname '_simmat_proteins_sw-n.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_go.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_mismatch-n-k3m1.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_mismatch-n-k3m2.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_mismatch-n-k4m1.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_mismatch-n-k4m2.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_spectrum-n-k3.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_spectrum-n-k4.txt'],...
                    ['data/kernels/' dataname '_simmat_proteins_ppi.txt'],...
                    };
        K1 = [];
        for i=1:length(k1_paths)
            [mat, labels] = loadtabfile(k1_paths{i});
            mat = process_kernel(mat);
            K1(:,:,i) = mat;
        end
        K1(:,:,i+1) = kernel_gip(y_train,1, 1);

        k2_paths = {['data/kernels/' dataname '_simmat_drugs_simcomp.txt'],...
                    ['data/kernels/' dataname '_simmat_drugs_lambda.txt'],...
                    ['data/kernels/' dataname '_simmat_drugs_marginalized.txt'],...
                    ['data/kernels/' dataname '_simmat_drugs_minmaxTanimoto.txt'],...
                    ['data/kernels/' dataname '_simmat_drugs_spectrum.txt'],...
                    ['data/kernels/' dataname '_simmat_drugs_tanimoto.txt'],...
                    ['data/kernels/' dataname '_simmat_drugs_aers-bit.txt'],...
                    ['data/kernels/' dataname '_simmat_drugs_aers-freq.txt'],...
                    ['data/kernels/' dataname '_simmat_drugs_sider.txt'],...
                    };
        K2 = [];
        for i=1:length(k2_paths)
            [mat, labels] = loadtabfile(k2_paths{i});
            mat = process_kernel(mat);
            K2(:,:,i) = mat;
        end
        K2(:,:,i+1) = kernel_gip(y_train,2, 1);

        %% perform predictions
        lambda = 1;
        regcoef = 0.25;
        
        [ y2 , alpha, beta ] = kronrls_mkl( K1, K2, y_train, lambda, regcoef );
        
        %% evaluate predictions
        yy=y;
        yy(yy==0)=-1;
        stats = evaluate_performance(y2(test_idx),yy(test_idx),'classification');

        fold_aupr = [fold_aupr, stats.aupr];
    end
    
    run_aupr(run,:)=fold_aupr;
end
mean(mean(run_aupr,2))