function [ data ] = process_kernel( data )
%PROCESS_KERNEL 
    % make kernel matrix symmetric
    k = (data+data')/2;
    % make kernel matrix PSD
    e = max(0, -min(eig(data)) + 1e-4);
    data = k + e*eye(length(data));
end

