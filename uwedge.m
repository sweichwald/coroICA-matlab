function [V, converged, iteration, meanoffdiag] = uwedge(Rx, varargin)
% uwedge
% Reference:
% Fast Approximate Joint Diagonalization Incorporating Weight Matrices.
% IEEE Transactions on Signal Processing, 2009.

p = inputParser;
addOptional(p, 'init', NaN);
addOptional(p, 'rm_x0', true);
addOptional(p, 'eps', 1e-10);
addOptional(p, 'n_iter_max', 1000);
addOptional(p, 'verbose', false);
addOptional(p, 'n_components', NaN);
parse(p, varargin{:});
params = p.Results;

% 0) Preprocessing

% Remove and remember 1st matrix
Rx1 = squeeze(Rx(1, :, :));
if params.rm_x0
    Rx = Rx(2:end, :, :);
end
[M, d, ~] = size(Rx);

if isnan(params.n_components)
    n_components = d;
else
    n_components = params.n_components;
end

% Initial guess
symmetrise = @(x) (squeeze(x) + squeeze(x)') ./ 2;
if any(any(isnan(params.init))) && n_components == d
    [H, E] = eig(symmetrise(Rx(1, :, :)), 'vector');
    V = diag(1 ./ sqrt(abs(E))')*H';
elseif any(any(isnan(params.init)))
    [H, E] = eig(symmetrise(Rx(1, :, :)), 'vector');
    mat = cat(2, diag(1 ./ sqrt(abs(E(1:n_components)))), zeros(n_components, d-n_components));
    V = mat*H';
else
    V = params.init(1:n_components, :);
end

V = bsxfun(@rdivide, V, sqrt(sum(V.^2, 2)));

for iteration = 1:params.n_iter_max
    % 1) Generate Rs
    Rs = zeros(size(Rx, 1), n_components, n_components);
    for k = 1:size(Rx, 1)
        Rs(k, :, :) =  V*squeeze(Rx(k, :, :))*V';
    end

    % 2) Use Rs to construct A, equation (24) in paper with W=Id
    % 3) Set A1=Id and substitute off-diagonals
    Rsdiag = zeros(size(Rs, 1), size(Rs, 2));
    for k = 1:size(Rs, 1)
        Rsdiag(k, :) = diag(squeeze(Rs(k, :, :)));
    end
    Rsdiagprod = Rsdiag'*Rsdiag;
    denom_mat = diag(Rsdiagprod)*diag(Rsdiagprod)' - Rsdiagprod.^2;
    Rkl = zeros(size(Rs, 2), size(Rs, 3));
    for k = 1:size(Rs, 1)
        Rkl = Rkl + diag(diag(squeeze(Rs(k, :, :)))) * squeeze(Rs(k, :, :));
    end
    Rkl = Rkl';
    num_mat = diag(diag(Rsdiagprod)) * Rkl - Rsdiagprod .* Rkl';
    denom_mat(denom_mat == 0) = eps;
    A = num_mat ./ (denom_mat + eye(n_components));
    A(1:(size(A, 1)+1):end) = 1;

    % 4) Set new V
    Vold = V;
    V = A \ Vold;

    % 5) Normalise V
    V = bsxfun(@rdivide, V, sqrt(sum(V.^2, 2)));

    % 6) Check convergence
    changeinV = max(abs(V(:) - Vold(:)));
    if iteration >= params.n_iter_max
        converged = false;
        break
    elseif changeinV < params.eps
        converged = true;
        break
    end
end

% Rescale
normaliser = diag(V*Rx1*V');
V = diag(1 ./ (sign(normaliser) .* sqrt(abs(normaliser)))) * V;
diagonals = zeros(size(Rx, 1), n_components, n_components);
sqoffdiags = nan(size(diagonals));
for k = 1:size(Rx, 1)
    diagonals(k, :, :) =  V*squeeze(Rx(k, :, :))*V';
    sqoffdiags(k, :, :) = diagonals(k, :, :) .^ 2;
    sqoffdiags(k, 1:size(sqoffdiags, 2)+1:end) = NaN;
end
meanoffdiag = nanmean(sqoffdiags(:));

end
