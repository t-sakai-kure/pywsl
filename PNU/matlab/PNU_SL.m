function [func_dec, outputs, funcs] = PNU_SL(x, y, prior, eta_list, options)
% PNU_SL  Estimates classifiers based on PNU learning
%   From samples
%     {x^P_i}^{n_P}_{i=1} ~ p(x|y=+1),
%     {x^N_i}^{n_N}_{i=1} ~ p(x|y=-1),
%     {x^U_i}^{n_U}_{i=1} ~ p(x) = \theta_P p(x|y=+1) + \theta_N p(x|y=-1),
%     \theta_P = p(y=+1), \theta_N = p(y=-1),
%   this code estimates a classifier
%     g(x): R^b -> R
%   based on PNU learning [1].
% 
% Input:
%   x: n by d sample matrix, contains positive, negative, and unlabeled
%     samples
%   y: n-dimensional vector
%      If y_i = +1 means x_i is labeled as positive
%      If y_i = -1 means x_i is labeled as negative
%      If y_i =  0 means x_i is unlabeled
%   prior: p(y=+1) class prior of unlabeled data
%   eta_list: candidates of \eta, [-1, 1]
%     If eta_list=-1, PNU_SL is reduced to NU learning
%     If eta_list= 0, PNU_SL is reduced to PN learning
%     If eta_list=+1, PNU_SL is reduced to PU learning
%   options.
%     n_fold:      the number of folds for cross-validation
%     model_type:  model for classifier
%       "gauss" uses Gaussian kernel basis function
%       "lm" uses linear basis function
%     lambda_list: candidates of regularization parameter
%     sigma_list:  candidates of bandwidth
%     b:           the number of basis functions
%     use_bias: If true, model with intercept is used
%       Example, model_type='lm', use_bias=true, the linear model w'*x+b is
%       used
%     etab: eta value for computing loss, i.e., when computing score 
%       for cross-validation, 
%       if etab >= 0, (1-etab)*R_PN(g) + etab*R_PU(g),
%       else, (1-etab)*R_PN(g) + etab*R_NU(g).
%
% Output:
%   func_dec: estimated classifier, whose input is n by d test sample matrix
%     and sign of the output is estimated class labels.
%   outputs.
%     w: estimated parameter of the model, if use_bias=true, w(end) is the
%     intercept of the model.
%   funcs: length(eta_list) by 1 cell array, 
%     estimated classifiers for different \eta
%
% Reference:
%   [1] T. Sakai, M.C. du Plessis, G. Niu, and M. Sugiyama
%     Semi-supervised classification based on classification from positive
%       and unlabeled data
%     arXiv:1605.06955, https://arxiv.org/abs/1605.06955
% 
% (c) Tomoya Sakai, The University of Tokyo, Japan.
%       sakai@ms.k.u-tokyo.ac.jp

narginchk(5, 5);
global model_type LOG;

assert(0 < prior && prior < 1);
assert(isequal(unique(y), [-1; 0; 1]));
assert(-1 <= min(eta_list)  && max(eta_list) <= 1);

n_fold      = get_field_with_default(options, 'n_fold',      5);
model_type  = get_field_with_default(options, 'model_type', 'gauss');
lambda_list = get_field_with_default(options, 'lambda_list', logspace(-3, 1, 10));
b           = get_field_with_default(options, 'n_basis',     200);
use_bias    = get_field_with_default(options, 'use_bias',    false);
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
        LOG.info(mfilename, 'Gauss kernel model is used.');        
    case 'lm'
        LOG.info(mfilename, 'Linear model is used.');        
    otherwise
        LOG.error(mfilename, 'Model type is invalid.');
        error('model type is invalid!\n');
end
n_eta    = length(eta_list);
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
        dp = xp;
        dn = xn;
        du = xu;
end
clear xp xn xu 

if strcmp(model_type, 'gauss')
    sigma_list = get_field_with_default(options, 'sigma_list', ...
        sqrt(median([dp(:); dn(:); du(:)]))*logspace(-2, 1, 10));    
else
    sigma_list = 1; % any value is ok
end
n_sigma  = length(sigma_list);

score_table = zeros(n_sigma, n_lambda, n_eta, n_fold);
if n_sigma == 1 && n_lambda == 1 && n_eta == 1
    score_table(1, 1, 1) = -inf;
else
    for ite_sigma = 1:n_sigma        
        sigma = sigma_list(ite_sigma);
        [Kp, Kn, Ku] = calc_ker(dp, dn, du, sigma, use_bias);
            
        for ite_fold = 1:n_fold                 
            Hp_tr = prior*(Kp(cv_index_p ~= ite_fold, :)'*Kp(cv_index_p ~= ite_fold, :)) ...
                /sum(cv_index_p ~= ite_fold);
            Hn_tr = (1-prior)*(Kn(cv_index_n ~= ite_fold, :)'*Kn(cv_index_n ~= ite_fold, :)) ...
                /sum(cv_index_n ~= ite_fold);
            Hu_tr = Ku(cv_index_u ~= ite_fold, :)'*Ku(cv_index_u ~= ite_fold, :) ...
                /sum(cv_index_u ~= ite_fold);
            hp_tr = prior*mean(Kp(cv_index_p ~= ite_fold, :), 1)';
            hn_tr = (1-prior)*mean(Kn(cv_index_n ~= ite_fold, :), 1)';
            hu_tr = mean(Ku(cv_index_u ~= ite_fold, :), 1)';
            Kp_te = Kp(cv_index_p == ite_fold, :);            
            Kn_te = Kn(cv_index_n == ite_fold, :);            
            Ku_te = Ku(cv_index_u == ite_fold, :);
            
            K_te = [ Kp_te; Kn_te; Ku_te];
            y_te = [ ones(size(Kp_te, 1), 1); 
                    -ones(size(Kn_te, 1), 1);
                    zeros(size(Ku_te, 1), 1)];
                        
            for ite_eta = 1:n_eta
                eta = eta_list(ite_eta);
                for ite_lambda = 1:n_lambda
                    lambda   = lambda_list(ite_lambda);

                    if strcmp(model_type, 'gauss')
                        LOG.trace(mfilename, sprintf('sigma: %f, fold: %d, eta: %.2f, lambda: %f', sigma, ite_fold, eta, lambda));
                    else
                        LOG.trace(mfilename, sprintf('fold: %d, eta: %.2f, lambda: %f', ite_fold, eta, lambda));
                    end
                    theta_cv = solve(Hp_tr, Hn_tr, Hu_tr, hp_tr, hn_tr, hu_tr, ...
                        lambda, eta, use_bias);                

                    if isempty(etab); mix = eta; else mix = etab; end
                    score_table(ite_sigma, ite_lambda, ite_eta, ite_fold) = ...
                        score_table(ite_sigma, ite_lambda, ite_eta, ite_fold) ...
                        + calc_loss(K_te*theta_cv, y_te, prior, mix);
                end % lambda
            end % eta
        end % hold
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
        [Kp, Kn, Ku] = calc_ker(dp, dn, du, sigma, use_bias);
        theta = solve(prior*(Kp'*Kp)/np, (1-prior)*(Kn'*Kn)/nn, (Ku'*Ku)/nu, ...
            prior*mean(Kp, 1)', (1-prior)*mean(Kn, 1)', mean(Ku, 1)', ...
            lambda, eta, use_bias);                
        funcs{ite_eta} = make_func(theta, xc, sigma, use_bias);   
    end
end
sigma  = sigma_list(best_sigma_index);
lambda = lambda_list(best_lambda_index);
eta    = eta_list(best_eta_index);
LOG.trace(mfilename, sprintf('score: %f\n', score_best));
LOG.trace(mfilename, sprintf('eta: %.2f', eta));
if strcmp(model_type, 'gauss')
    LOG.trace(mfilename, sprintf('selected sigma=%.4f, lambda=%.4f\n', sigma, lambda));
else
    LOG.trace(mfilename, sprintf('selected lambda=%.4f\n', lambda));
end
[Kp, Kn, Ku] = calc_ker(dp, dn, du, sigma, use_bias);
theta = solve(prior*(Kp'*Kp)/np, (1-prior)*(Kn'*Kn)/nn, (Ku'*Ku)/nu, ...
    prior*mean(Kp, 1)', (1-prior)*mean(Kn, 1)', mean(Ku, 1)', ...
    lambda, eta, use_bias);                
func_dec = make_func(theta, xc, sigma, use_bias);

if nargout > 1
    outputs.sigma_index  = best_sigma_index;    
    outputs.lambda_index = best_lambda_index;
    outputs.eta_index  = best_eta_index;
    outputs.score_table  = score_table;    
    outputs.score_list   = score_list;
    outputs.w            = theta;   
end

end

function theta = solve(Hp, Hn, Hu, hp, hn, hu, lambda, eta, use_bias)

if isnan(hp) % if np = 0;
    Hp = 0; 
    hp = 0; 
end
    
if isnan(hn) % if nn = 0;
    Hn = 0; 
    hn = 0; 
end

if isnan(hu) % if nu = 0;
    Hu = 0;
    hu = 0;
end

b   = size(Hu, 1);
Reg = lambda*eye(b); 
if use_bias
    Reg(b, b) = 0;
end

Hpn = Hp + Hn;
hpn = hp - hn;

if eta >= 0 % PNPU
    hpu = 2*hp - hu;
    R     = chol((1-eta)*Hpn + eta*Hu + Reg); 
    theta = R\(R'\((1-eta)*hpn + eta*hpu));
else % PNNU
    hnu = hu - 2*hn;
    eta = -eta;
    R     = chol((1-eta)*Hpn + eta*Hu + Reg);
    theta = R\(R'\((1-eta)*hpn + eta*hnu));
end

end

function loss = calc_loss(g, y, prior, etab)
% calculate the loss

np = sum(y ==  1);
nn = sum(y == -1);
nu = sum(y ==  0);

if np ~= 0; fn   = mean(g(y ==  1) <= 0); else fn   = 0; end
if nn ~= 0; fp   = mean(g(y == -1) >= 0); else fp   = 0; end
loss_pn = prior*fn + (1 - prior)*fp;

if etab >= 0 % PNPU
    if nu ~= 0; fp_u = mean(g(y ==  0) >= 0); else fp_u = 0; end
    loss_pu = prior*fn + max(fp_u + prior*fn - prior, 0);
    loss    = (1-etab)*loss_pn + etab*loss_pu;
else % PNNU
    etab  = -etab;
    if nu ~= 0; fn_u = mean(g(y ==  0) <= 0); else fn_u = 0; end
    loss_nu = max(fn_u + (1 - prior)*fp - (1 - prior), 0) + (1 - prior)*fp;
    loss    = (1-etab)*loss_pn + etab*loss_nu;
end

end


function func_dec = make_func(theta, xc, sigma, use_bias)
global model_type

switch model_type
    case 'gauss'
        if use_bias
            func_dec = @(x_test) [exp(-calc_dist2(x_test, xc)/(2*sigma^2)), ones(size(x_test, 1), 1)]*theta;
        else
            func_dec = @(x_test) exp(-calc_dist2(x_test, xc)/(2*sigma^2))*theta;
        end
    case 'lm'
        if use_bias
            func_dec = @(x_test) [x_test, ones(size(x_test, 1), 1)]*theta;
        else
            func_dec = @(x_test) x_test*theta;
        end
end

end


function [Kp, Kn, Ku] = calc_ker(dp, dn, du, sigma, use_bias)
global model_type;

np = size(dp, 1);
nn = size(dn, 1);
nu = size(du, 1);

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
if use_bias
    Kp = [Kp, ones(np, 1)];
    Kn = [Kn, ones(nn, 1)];
    Ku = [Ku, ones(nu, 1)];
end

end

function dist2 = calc_dist2(x, xc)
% make n by b squared-distance matrix, 
%   n is the number of samples, b is the number of basis functions.

dist2 = bsxfun(@plus, sum(x.^2, 2), bsxfun(@minus, sum(xc.^2, 2)', 2*x*xc'));

end


function ret = get_field_with_default(field, name, default)

if ~isfield(field, name) || isempty(field.(name));
    field.(name) = default;
end
ret = field.(name);

end