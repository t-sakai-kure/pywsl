function [xp_tr, xn_tr, xu_tr, x_te, y_te] = load_ringnorm(np, nn, nu, prior)

pc =  1;
nc = -1;

load('ringnorm_data.mat');

[xp_tr, xn_tr, xu_tr, x_te, y_te] = split_data(data', label', pc, nc, np, nn, nu, prior);

end