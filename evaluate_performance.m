function stats = evaluate_performance(dec, labels, type);
% Modified version of the validation_function() included in libSVM 
% do_binary_predict matlab package.

if nargin < 3, type = 'classification'; end

% labels = (labels >= 0) - (labels < 0);
stats = {};

if isequal(type,'classification')
    stats.precision = precision(dec, labels);
    stats.recall    = recall(dec, labels);
    stats.sensitivity = stats.recall;
    stats.specificity = specificity(dec, labels);
    stats.fscore    = fscore(dec, labels);
    stats.bac       = bac(dec, labels);
    stats.auc       = auc(dec, labels);
    stats.accuracy  = accuracy(dec, labels);
    stats.aupr      = aupr(dec,labels);
    stats.ci        = concordance_index(dec, labels); 
else
    stats.precision = 0;
    stats.recall    = 0;
    stats.sensitivity = 0;
    stats.specificity = 0;
    stats.fscore    = 0;
    stats.bac       = 0;
    stats.auc       = 0;
    stats.accuracy  = 0;
    stats.aupr      = 0;
    stats.ci        = concordance_index(dec, labels);
end

function ret = precision(dec, label)
tp = sum(label == 1 & dec >= 0);
tp_fp = sum(dec >= 0);
if tp_fp == 0;
  disp(sprintf('warning: No positive predict label.'));
  ret = 0;
else
  ret = tp / tp_fp;
end
% disp(sprintf('Precision = %g%% (%d/%d)', 100.0 * ret, tp, tp_fp));

function ret = recall(dec, label)
tp = sum(label == 1 & dec >= 0);
tp_fn = sum(label == 1);
if tp_fn == 0;
  disp(sprintf('warning: No postive true label.'));
  ret = 0;
else
  ret = tp / tp_fn;
end
% disp(sprintf('Recall = %g%% (%d/%d)', 100.0 * ret, tp, tp_fn));

function ret = specificity(dec, label)
tn = sum(label == -1 & dec < 0);
tn_fp = sum(label == -1);
if tn_fp == 0;
  disp(sprintf('warning: No negative true label.'));
  specificity = 0;
else
  specificity = tn / tn_fp;
end
ret = specificity;


function ret = fscore(dec, label)
tp = sum(label == 1 & dec >= 0);
tp_fp = sum(dec >= 0);
tp_fn = sum(label == 1);
if tp_fp == 0;
  disp(sprintf('warning: No positive predict label.'));
  precision = 0;
else
  precision = tp / tp_fp;
end
if tp_fn == 0;
  disp(sprintf('warning: No postive true label.'));
  recall = 0;
else
  recall = tp / tp_fn;
end
if precision + recall == 0;
  disp(sprintf('warning: precision + recall = 0.'));
  ret = 0;
else
  ret = 2 * precision * recall / (precision + recall);
end
% disp(sprintf('F-score = %g', ret));

function ret = bac(dec, label)
tp = sum(label == 1 & dec >= 0);
tn = sum(label == -1 & dec < 0);
tp_fn = sum(label == 1);
tn_fp = sum(label == -1);
if tp_fn == 0;
  disp(sprintf('warning: No positive true label.'));
  sensitivity = 0;
else
  sensitivity = tp / tp_fn;
end
if tn_fp == 0;
  disp(sprintf('warning: No negative true label.'));
  specificity = 0;
else
  specificity = tn / tn_fp;
end
ret = (sensitivity + specificity) / 2;
% disp(sprintf('BAC = %g', ret));

function ret = auc(dec, label)
[dec idx] = sort(dec, 'descend');
label = label(idx);
tp = cumsum(label == 1);
fp = sum(label == -1);
ret = sum(tp(label == -1));
if tp == 0 | fp == 0;
  disp(sprintf('warning: Too few postive true labels or negative true labels'));
  ret = 0;
else
  ret = ret / tp(end) / fp;
end
%   disp(sprintf('AUC = %g', ret));

function ret = accuracy(dec, label)
  correct = sum(dec .* label >= 0);
  total = length(dec);
  ret = correct / total;
%   disp(sprintf('Accuracy = %g%% (%d/%d)', 100.0 * ret, correct, total));

function aupr = aupr(dec, label)
    % Calculate area under the Precission/recall curve
    [Xpr,Ypr,Tpr,aupr] = perfcurve(label, dec, 1, 'xCrit', 'reca', 'yCrit', 'prec'); 
    
function ci = concordance_index(dec, label)
    % REF: https://stat.ethz.ch/pipermail/r-help/2008-December/182565.html
    % 1) Create all pairs of observed responses.
    % 2) For all valid response pairs, i.e., pairs where one response y_1 is 
    % greater than the other y_2, test whether the corresponding predictions 
    % are concordant, i.e, yhat_1 > yhat_2. If so add 1 to the running sum s. 
    % If yhat_1 = yhat_2, add 0.5 to the sum. Count the number n of valid 
    % response pairs.
    % 3) Divide the total sum s by the number of valid response pairs n

    n = length(label);
    total  = 0;
    
    % \sum_{i=1}^{n-1} (n-i) = (1/2) * n *(n-1)
    % ref: http://www.wolframalpha.com/input/?i=sum+n-i%2C+i%3D1+to+n-1
    norm_z = 0;
    for j=1:n-1
        idx   = find(label(j) > label);
        pairs = idx(find(idx ~= j));
        norm_z = norm_z + length(pairs);        
        total = total + sum(dec(j) > dec(pairs)) + sum(dec(j)==dec(pairs))*0.5;        
    end
    ci = total / norm_z;    
