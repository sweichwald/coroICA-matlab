function [V, converged, n_iter, meanoffdiag] = coroICA(X, varargin)
% coroICA    implements the coroICA algorithm presented in
% Robustifying Independent Component Analysis by Adjusting for Group-Wise Stationary Noise
% N Pfister*, S Weichwald*, P Bühlmann, B Schölkopf
% https://arxiv.org/abs/1806.01094
%
% [V, converged, n_iter, meanoffdiag] = coroICA(X, varargin)
%
% Required argument
% -----------
%     X : array, shape (n_samples, n_features)
%         where n_samples is the number of samples and
%         n_features is the number of features.
%
% Optional arguments coroICA(X, 'optionalarg', optionalargval, ...)
% -----------
%    group_index : array, optional, shape (n_samples, 1)
%         Codes for each sample which group it belongs to; if no group index
%         is provided a rigid grid with groupsize samples per
%         group is used (which defaults to all samples if groupsize
%         was not set).
%    partition_index : array, optional, shape (n_samples, 1)
%         Codes for each sample which partition it belongs to; if no
%         partition index is provided a rigid grid with partitionsize
%         samples per partition within each group is used (which has a
%         (hopefully sane) default if partitionsize was not set).
%    n_components : int, optional
%         Number of components to extract. If NaN is passed, the same number of
%         components as the input has dimensions is used.
%    n_components_uwedge : int, optional
%         Number of components to extract during uwedge approximate joint
%         diagonalization of the matrices. If NaN is passed, the same number of
%         components as the input has dimensions is used.
%    rank_components : boolean, optional
%         When true, the components will be ordered in decreasing stability.
%    pairing : {'complement', 'allpairs'}
%         Whether difference matrices should be computed for all pairs of
%         partition covariance matrices or only in a one-vs-complement scheme.
%    groupsize : int, optional
%         Approximately how many samples, when doing a rigid grid, shall be in
%         each group. If NaN is passed, all samples will be in one group unless
%         group_index is passed in which case the provided group
%         index is used (the latter is the advised and preferred way).
%    partitionsize : int, optional
%         Approxiately how many samples, when doing a rigid grid, should be in
%         each partition. If NaN is passed, a (hopefully sane) default is used,
%         again, unless partition_index is passed in which case
%         the provided partition index is used.
%    max_iter : int, optional
%         Maximum number of iterations for the uwedge approximate joint
%         diagonalisation during fitting.
%    tol : float, optional
%         Tolerance for terminating the uwedge approximate joint diagonalisation
%         during fitting.
%
% Output
% -----------
%    V : array, shape (n, n_features)
%        The unmixing matrix; where n=n_features if n_components and
%        n_components_uwedge are NaN, n=n_components_uwedge if n_components is
%        NaN, and n=n_components otherwise.
%    converged : boolean
%        Whether the approximate joint diagonalisation converged due to tol.
%    n_iter : int
%        Number of iterations of the approximate joint diagonalisation.
%    meanoffdiag : float
%        Mean absolute value of the off-diagonal values of the to be jointly
%        diagonalised matrices, i.e., a proxy of the approximate joint
%        diagonalisation objective function.
%
% Example usage
% -----------
% [V, converged, n_iter, meanoffdiag] = coroICA(Xtrain, 'group_index', group_index, 'partition_index', partition_index);
% transformed_samples = (V * Xtest')';

p = inputParser;
addOptional(p, 'group_index', NaN);
addOptional(p, 'partition_index', NaN);
addOptional(p, 'n_components', NaN);
addOptional(p, 'n_components_uwedge', NaN);
addOptional(p, 'rank_components', false);
addOptional(p, 'pairing', 'complement');
addOptional(p, 'groupsize', NaN);
addOptional(p, 'partitionsize', NaN);
addOptional(p, 'max_iter', 1000);
addOptional(p, 'tol', 1e-12);
parse(p, varargin{:});
params = p.Results;

[n, dim] = size(X);

% generate group and partition indices as needed
if any(any(isnan(params.group_index))) && isnan(params.groupsize)
    group_index = zeros(n, 1);
elseif any(any(isnan(params.group_index)))
    group_index = rigidgroup(n, params.groupsize);
else
    group_index = params.group_index;
end

if any(any(isnan(params.partition_index))) && isnan(params.partitionsize)
    smallest_group = n;
    for group = unique(group_index)'
        smallest_group = min(smallest_group, sum(group_index == group));
    end
    partition_index = rigidpartition(group_index, max(dim, smallest_group / 2));
elseif any(any(isnan(params.partition_index)))
    partition_index = rigidpartition(group_index, params.partitionsize);
else
    partition_index = params.partition_index;
end

X = X';

no_groups = length(unique(group_index));

% computing covariance difference matrices
if strcmp(params.pairing, 'complement')
    covmats = zeros(0, dim, dim);
    idx = 1;
    for group = unique(group_index)'
        for partition = unique(partition_index(group_index == group))'
            ind1 = ((partition_index == partition) & (group_index == group));
            ind2 = ((partition_index ~= partition) & (group_index == group));
            if sum(ind2) > 0
                covmats(idx, :, :) = cov(X(:, ind1)') - cov(X(:, ind2)');
                idx = idx + 1;
            else
                warning(['Removing group ', num2str(group), ' since the partition is trivial, i.e., contains only exactly one set.']);
            end
        end
    end
elseif strcmp(params.pairing, 'allpairs')
    subvec = zeros(no_groups, 1);
    unique_group_index = unique(group_index);
    for i = 1:length(unique_group_index)
        group = unique_group_index(i);
        subvec(i) = length(unique(partition_index(group_index == group)));
    end
    covmats = zeros(0, dim, dim);
    idx = 1;
    for count = 1:length(unique_group_index)
        group = unique_group_index(count);
        unique_subs = unique(partition_index(group_index == group));
        if subvec(count) <= 1
            warning(['Removing group ', num2str(group), ' since the partition is trivial, i.e., contains only exactly one set.']);
        else
            for i = 1:(subvec(count) - 1)
                for j = (i + 1):subvec(count)
                    ind1 = ((partition_index == unique_subs(i)) & (group_index == group));
                    ind2 = ((partition_index == unique_subs(j)) & (group_index == group));
                    covmats(idx, :, :) = cov(X(:, ind1)') - cov(X(:, ind2)');
                    idx = idx + 1;
                end
            end
        end
    end
end

% add total observational covariance for normalization
covmats = cat(1, zeros(1, dim, dim), covmats);
covmats(1, :, :) = cov(X');

% joint diagonalisation
[V, converged, n_iter, meanoffdiag] = uwedge(covmats, ...
                                             'rm_x0', true, ...
                                             'eps', params.tol, ...
                                             'n_iter_max', params.max_iter, ...
                                             'n_components', params.n_components_uwedge)

% rank components
if params.rank_components || ~isnan(params.n_components)
    A = pinv(V);
    dimU = size(A, 2);
    colcorrs = zeros(dimU, 1);
    % running average
    for k = 1:size(covmats, 1)
        tmpmat = abs(corrcoef(cat(1, A', V*squeeze(covmats(k, :, :))')'));
        colcorrs = colcorrs + diag(tmpmat(1:dimU, (dimU+1):end)) ./ size(covmats, 1);
    end
    [~, sorting] = sort(-colcorrs);
    V = V(sorting, :);
end

if ~isnan(params.n_components)
    V = V(1:n_components, :);
end

end


function partition = rigidpartition(group, nosamples)
partition = zeros(size(group));
for e = unique(group)'
    partition(group == e) = rigidgroup(sum(group == e), nosamples) + max(partition) + 1;
end
end


function index = rigidgroup(length, nosamples)
groups = floor(length / nosamples);
changepoints = [];
for a = linspace(1, length + 1, groups + 1)
    changepoints(end+1) = round(a);
end
changepoints = unique(changepoints);
changepoints = sort(changepoints);
index = zeros(length, 1);
for i = 1:groups
    index(changepoints(i):changepoints(i+1)-1) = i;
end
end
