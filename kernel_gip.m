function d = kernel_gip(adjmat,dim, gamma)
% Calculates the Gaussian Interaction Profile (Laarhoven, 2011) kernel from a graph adjacency 
% matrix. If tha graph is unipartite, ka = kb.
%INPUT: 
% adjmat : binary adjacency matrix
% dim    : dimension (1 - rows, 2 - cols)
%OUTPUT:
% d : kernel matrix for adjmat over dimension 'dim'

    y = adjmat;
    % Graph based kernel
	if dim == 1
        ga = y*y';
    else
        ga = y'*y;
    end
	ga = gamma * ga / mean(diag(ga));
	d = exp(-kernel_to_distance(ga)); 
	
end

function d=kernel_to_distance(k)
    %Based on Phillips2011 and
    % Improved Graph-Based Metrics for Clustering High-Dimensional Datasets
    % || x - z ||^2 --> <x,x> + <z,z> + 2<x,z>  
    
	% Given a kernel matrix k=y*y', calculate a matrix of square Euclidean distances
	di = diag(k);
	d = repmat(di,1,length(k)) + repmat(di',length(k),1) - 2 * k;
end