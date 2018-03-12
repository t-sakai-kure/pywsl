function [func_dec, outputs, funcs] = PNU_LL(x, y, prior, eta_list, options)
narginchk(5, 5);
global model_type LOG;

assert(0 < prior & prior < 1);
assert(isequal(unique(y), [-1; 0; 1]));
assert(-1 <= min(eta_list)  && max(eta_list) <= 1);

n_fold      = get_field_with_default(options, 'n_fold',      5);
model_type  = get_field_with_default(options, 'model_type', 'gauss');
lambda_list = get_field_with_default(options, 'lambda_list', logspace(-3, 1, 10));
b           = get_field_with_default(options, 'n_basis',     200);
etab        = get_field_with_default(options, 'etab',        []);

xp = x(y ==  1, :);
xn = x(y == -1, :);
xu = x(y ==  0, :);
    
np = sum(y ==  1);
nn = sum(y == -1);
nu = sum(y ==  0);

xl = x(y ~= 0, :);
bl = min(b, np + nn);
bu = max(min(b - bl, nu), 0);
center_index_l = randperm(np + nn, bl);
center_index_u = randperm(nu, bu);
xc = [xl(center_index_l, :); xu(center_index_u, :)];

model_type = lower(model_type);
switch model_type
    case 'gauss'
        LOG.info(mfilename, 'Gauss kernel is used.');        
    case 'lm'
        LOG.info(mfilename, 'Linear model is used.');        
    otherwise
        LOG.error(mfilename, 'kernel type is invalid.');
        error('no kernel type\n');
end
n_eta  = length(eta_list);
n_lambda = length(lambda_list);

cv_index_p = floor((0:(np - 1))*n_fold/np) + 1;
cv_index_p = cv_index_p(randperm(np));
cv_index_n = floor((0:(nn - 1))*n_fold/nn) + 1;
cv_index_n = cv_index_n(randperm(nn));
cv_index_u = floor((0:(nu - 1))*n_fold/nu) + 1;
cv_index_u = cv_index_u(randperm(nu));

switch model_type
    case 'gauss'
        dp = calc_dist2(xp, xc);
        dn = calc_dist2(xn, xc);
        du = calc_dist2(xu, xc);
    case 'lm'
%         dp = xp;
%         dn = xn;
%         du = xu;
        dp = [xp, ones(np, 1)];
        dn = [xn, ones(nn, 1)];
        du = [xu, ones(nu, 1)];        
end

if strcmp(model_type, 'gauss')
    sigma_list = get_field_with_default(options, 'sigma_list', ...
        sqrt(median([dp(:); dn(:); du(:)]))*logspace(-2, 1, 10));    
else
    sigma_list = 1; % any value is ok
end
n_sigma  = length(sigma_list);

score_table = zeros(n_sigma, n_lambda, n_eta, n_fold);
if n_sigma == 1 && n_lambda == 1 && n_eta == 1
    score_table(1, 1, 1, 1) = -inf;
else
    for ite_sigma = 1:n_sigma        
        sigma = sigma_list(ite_sigma);
        [Kp, Kn, Ku] = calc_ker(dp, dn, du, sigma);
            
        for ite_fold = 1:n_fold     
            Kp_tr = Kp(cv_index_p ~= ite_fold, :);
            Kp_te = Kp(cv_index_p == ite_fold, :);
            Kn_tr = Kn(cv_index_n ~= ite_fold, :);
            Kn_te = Kn(cv_index_n == ite_fold, :);
            Ku_tr = Ku(cv_index_u ~= ite_fold, :);
            Ku_te = Ku(cv_index_u == ite_fold, :);
            
            for ite_eta = 1:n_eta
                eta = eta_list(ite_eta);
                for ite_lambda = 1:n_lambda
                    lambda   = lambda_list(ite_lambda);

                    if strcmp(model_type, 'gauss')
                        LOG.trace(mfilename, sprintf('sigma: %f, lambda: %f', sigma, lambda));
                    else
                        LOG.trace(mfilename, sprintf('lambda: %f', lambda));
                    end
                    [K_tr, y_tr, z_tr] = agg(Kp_tr, Kn_tr, Ku_tr, eta);
                    w_cv = solve(K_tr, y_tr, z_tr, prior, lambda, eta);                                    

                    if isempty(etab); mix = eta; else mix = etab; end
                    [K_te, y_te]       = agg(Kp_te, Kn_te, Ku_te, mix);                    
                    score_table(ite_sigma, ite_lambda, ite_eta, ite_fold) = ...
                        score_table(ite_sigma, ite_lambda, ite_eta, ite_fold) ...
                        + calc_loss(K_te*w_cv, y_te, prior, mix);
                end % lambda
            end % gamma
        end % fold
    end % sigma
    score_table = mean(score_table, 4);
end
score_list = zeros(n_eta, 1);
score_best = inf;
funcs      = cell(n_eta, 1);
for ite_eta = 1:n_eta
    sub_score_table   = score_table(:, :, ite_eta);
    [~, chosen_index] = min(sub_score_table(:));    
    [sigma_index, lambda_index] = ind2sub(size(sub_score_table), chosen_index);
    score_list(ite_eta)       = sub_score_table(chosen_index);    
    if score_list(ite_eta) < score_best
        score_best  = score_list(ite_eta);
        best_sigma_index  = sigma_index;
        best_lambda_index = lambda_index;
        best_eta_index  = ite_eta;                
    end
    if nargout > 2
        sigma  = sigma_list(sigma_index);
        lambda = lambda_list(lambda_index);
        eta  = eta_list(ite_eta);
        [Kp, Kn, Ku] = calc_ker(dp, dn, du, sigma);
        [K, y, z]    = agg(Kp, Kn, Ku, eta);
        w = solve(K, y, z, prior, lambda, eta);                
        funcs{ite_eta} = make_func(w, xc, sigma);   
    end
end
sigma  = sigma_list(best_sigma_index);
lambda = lambda_list(best_lambda_index);
eta  = eta_list(best_eta_index);
LOG.trace(mfilename, sprintf('score: %f\n', score_best));
if strcmp(model_type, 'gauss')
    LOG.trace(mfilename, sprintf('selected sigma=%.4f, lambda=%.4f\n', sigma, lambda));
else
    LOG.trace(mfilename, sprintf('selected lambda=%.4f\n', lambda));
end

[Kp, Kn, Ku] = calc_ker(dp, dn, du, sigma);
[K, y, z]    = agg(Kp, Kn, Ku, eta);
w            = solve(K, y, z, prior, lambda, eta); 
func_dec     = make_func(w, xc, sigma);

if nargout > 1
    outputs.sigma_index  = best_sigma_index;    
    outputs.lambda_index = best_lambda_index;
    outputs.eta_index    = best_eta_index;
    outputs.score_table  = score_table;    
    outputs.score_list   = score_list;
    outputs.w            = w;
end

end


function w = solve(K, y, z, prior, lambda, eta)

mfun_opts = [];
mfun_opts.maxFunEvals = 500; %1000;   
mfun_opts.maxIter     = 250; %500;

b   = size(K, 2);
fun = @(w) derivative(w, K, y, z, prior, lambda, eta);
w   = minFunc(fun, zeros(b, 1), mfun_opts);

end


function [val, der] = derivative(w, K, y, z, prior, lambda, eta)

pnpu = true;
if eta < 0 % switch PNNU
    pnpu = false;
    eta = -eta;
end

np = sum(y ==  1);
nn = sum(y == -1);
nu = sum(y ==  0);
n  = np + nn + nu;

if np ~= 0; ap = eta*prior/np; else ap = 0; end
if nn ~= 0; an = eta*(1 - prior)/nn; else an = 0; end
if np ~= 0; cp = (1 - eta)*prior/np; else cp = 0; end
if nn ~= 0; cn = (1 - eta)*(1 - prior)/nn; else cn = 0; end
if nu ~= 0; cu = eta/nu; else cu = 0; end

c   = cp.*(y == 1) + cn.*(y == -1) + cu.*(y == 0);
lmd = lambda*ones(length(w), 1)/n;

zKw = z.*(K*w);

if pnpu
    val = max(-ap*sum(K(y == 1, :), 1)*w + sum(c.*logsumexp([zeros(n, 1), -zKw])) ...
        + (lmd.*w)'*w/2, 0);
    der_lin = -ap*sum(K(y == 1, :), 1)';
else
    val = max(an*sum(K(y == -1, :), 1)*w + sum(c.*logsumexp([zeros(n, 1), -zKw])) ...
        + (lmd.*w)'*w/2, 0);
    der_lin = an*sum(K(y == -1, :), 1)';    
end

der_log = -sum(bsxfun(@times, c.*z.*sig(-zKw), K), 1)';
der     = der_lin + der_log + lmd.*w;

end


function loss = calc_loss(g, y, prior, eta)
% calculate the objective

np = sum(y ==  1);
nn = sum(y == -1);
nu = sum(y ==  0);

if np ~= 0; fn   = mean(g(y ==  1) <= 0); else fn   = 0; end
if nn ~= 0; fp   = mean(g(y == -1) >= 0); else fp   = 0; end
loss_pn = prior*fn + (1 - prior)*fp;

if eta >= 0
    if nu ~= 0; fp_u = mean(g(y ==  0) >= 0); else fp_u = 0; end
    loss_pu = prior*fn + max(fp_u + prior*fn - prior, 0);
    % loss_pu = 2*prior*fn + fp_u - prior;
    loss    = (1 - eta)*loss_pn + eta*loss_pu;
else
    eta  = -eta;
    if nu ~= 0; fn_u = mean(g(y ==  0) <= 0); else fn_u = 0; end
    loss_nu = max(fn_u + (1 - prior)*fp - (1 - prior), 0) + (1 - prior)*fp;
    % loss_nu = fn_u + 2*(1 - prior)*fp - (1 - prior);
    loss    = (1 - eta)*loss_pn + eta*loss_nu;
end

end


function [x, y, z] = agg(xp, xn, xu, eta)

np = size(xp, 1);
nn = size(xn, 1);
nu = size(xu, 1);

if eta == 0 % PN
    x = [xp; xn];
    y = [ones(np, 1); -ones(nn, 1)];
    z = [ones(np, 1); -ones(nn, 1)];    
elseif eta == 1 % PU
    x = [xp; xu];
    y = [ones(np, 1); zeros(nu, 1)];
    z = [ones(np, 1); -ones(nu, 1)];
elseif eta == -1 % NU
    x = [xn; xu];
    y = [-ones(nn, 1); zeros(nu, 1)];
    z = [-ones(nn, 1);  ones(nu, 1)];        
elseif 0 < eta && eta < 1 % PNPU
    x = [xp; xn; xu];
    y = [ones(np, 1); -ones(nn, 1); zeros(nu, 1)];
    z = [ones(np, 1); -ones(nn, 1); -ones(nu, 1)];        
else % PNNU
    x = [xp; xn; xu];
    y = [ones(np, 1); -ones(nn, 1); zeros(nu, 1)];
    z = [ones(np, 1); -ones(nn, 1);  ones(nu, 1)];        
end

end


function lse = logsumexp(x)

b   = max(x, [], 2);
lse = b + log(sum(exp(bsxfun(@minus, x, b)), 2));

end


function sig = sig(x)
% x: d-dimensional vector
assert(size(x, 2) == 1);

sig = zeros(size(x));

idx = x >= 0;

sig(idx) = 1./(1 + exp(-x(idx)));
sig(~idx) = exp(x(~idx))./(1 + exp(x(~idx)));

end


function func_dec = make_func(theta, xc, sigma)
global model_type

switch model_type
    case 'gauss'
        func_dec = @(x_test) exp(-calc_dist2(x_test, xc)/(2*sigma^2))*theta;
    case 'lm'
        func_dec = @(x_test) [x_test, ones(size(x_test, 1), 1)]*theta;
end

end


function [Kp, Kn, Ku] = calc_ker(dp, dn, du, sigma)
global model_type;

switch model_type
    case 'gauss'
        Kp = exp(-dp/(2*sigma^2));
        Kn = exp(-dn/(2*sigma^2));
        Ku = exp(-du/(2*sigma^2));
    case 'lm'
        Kp = dp;
        Kn = dn;
        Ku = du;
end

end


function dist2 = calc_dist2(x, xc)
% make n by b squared-distance matrix, 
%   n is the number of samples, b is the number of basis functions.

dist2 = bsxfun(@plus, sum(x.^2, 2), bsxfun(@minus, sum(xc.^2, 2)', 2*x*xc'));

end


function ret = get_field_with_default(field, name, default)

if ~isfield(field, name) || isempty(getfield(field, name)); %#ok<GFLD>
    field = setfield(field, name, default); %#ok<SFLD>
end
ret = getfield(field, name); %#ok<GFLD>

end


