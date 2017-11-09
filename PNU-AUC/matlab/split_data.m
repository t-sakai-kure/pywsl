function [xp_tr, xn_tr, xu_tr, x_te, y_te] = ...
    split_data(data, labels, pc, nc, np, nn, nu, prior)

p_data   = data(labels == pc, :);
n_data   = data(labels == nc, :);
np_all   = size(p_data, 1);
nn_all   = size(n_data, 1);

p_idx_tr = randperm(np_all, np);
n_idx_tr = randperm(nn_all, nn);
xp_tr    = p_data(p_idx_tr, :);
xn_tr    = n_data(n_idx_tr, :);
p_data(p_idx_tr, :) = [];
n_data(n_idx_tr, :) = [];
np_all   = size(p_data, 1);
nn_all   = size(n_data, 1);

yu       = 1 - 2*(rand(nu, 1) > prior);
nu_p     = sum(yu == +1);
nu_n     = sum(yu == -1);
p_idx_tr = randperm(np_all, nu_p);
n_idx_tr = randperm(nn_all, nu_n);
xu_p_tr  = p_data(p_idx_tr, :);
xu_n_tr  = n_data(n_idx_tr, :);
xu_tr    = [xu_p_tr; xu_n_tr];
p_data(p_idx_tr, :) = [];
n_data(n_idx_tr, :) = [];
np_all   = size(p_data, 1);
nn_all   = size(n_data, 1);

% if np_all < 100 || nn_all < 100
if np_all < 17 || nn_all < 100
    fprintf('np_all: %d, nn_all: %d\n', np_all, nn_all);
    error('test data is too small');
end
nt_p     = min(1000, np_all);
nt_n     = min(1000, nn_all);
p_idx_te = randperm(np_all, nt_p);
n_idx_te = randperm(nn_all, nt_n);
xp_te    = p_data(p_idx_te, :);
xn_te    = n_data(n_idx_te, :);
x_te     = [xp_te; xn_te];
y_te     = [ones(nt_p, 1); -ones(nt_n, 1)];

end
