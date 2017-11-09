function [f_dec, f_auc, outputs] = PNU_AUC_SL(x, y, prior, eta_list, options)
global ker_type LOG;

ker_type    = get_field_with_default(options, 'kernel_type', 'gauss');
lambda_list = get_field_with_default(options, 'lambda_list', logspace(-3, 1, 10));
sigma_list  = get_field_with_default(options, 'sigma_list',  logspace(-3, 1, 10));
b           = get_field_with_default(options, 'n_basis',     200);
n_fold      = get_field_with_default(options, 'n_fold',      5);

np = sum(y == +1);
nn = sum(y == -1);
nu = sum(y ==  0);
n  = length(y);

xp = x(y == +1, :);
xn = x(y == -1, :);
xu = x(y ==  0, :);

b = min(b, n);
center_index = randperm(n, b);
xc = x(center_index, :); 

ker_type = lower(ker_type);
switch ker_type
    case 'gauss'
        LOG.info(mfilename, 'Gauss kernel is used.');        
    case 'lm'
        LOG.info(mfilename, 'Linear model is used.');        
        sigma_list = 0;        
    case 'linear'
        LOG.info(mfilename, 'Linear kernel is used.');        
        sigma_list = 0;                
    otherwise
        LOG.error(mfilename, 'kernel type is invalid.');
        error('no kernel type\n');
end
n_lambda = length(lambda_list);
n_sigma  = length(sigma_list);
n_eta    = length(eta_list);

cv_index_p = floor((0:(np - 1))*n_fold/np) + 1;
cv_index_p = cv_index_p(randperm(np));
cv_index_n = floor((0:(nn - 1))*n_fold/nn) + 1;
cv_index_n = cv_index_n(randperm(nn));
cv_index_u = floor((0:(nu - 1))*n_fold/nu) + 1;
cv_index_u = cv_index_u(randperm(nu));

switch ker_type
    case 'gauss'
        dp = calc_dist2(xp, xc);
        dn = calc_dist2(xn, xc);
        du = calc_dist2(xu, xc);        
    case 'lm'
        dp = xp;
        dn = xn;
        du = xu;
    case 'linear'
        dp = xp*xc';
        dn = xn*xc';
        du = xu*xc';        
end

score_table = zeros(n_sigma, n_lambda, n_eta);
if n_sigma == 1 && n_lambda == 1 && n_eta == 1
    score_table(1, 1, 1) = -inf;
else
    for ite_sigma = 1:n_sigma        
        sigma = sigma_list(ite_sigma);
        [Kp, Kn, Ku] = calc_ker(dp, dn, du, sigma);
            
        for ite_fold = 1:n_fold
            Kp_tr = Kp(cv_index_p ~= ite_fold, :);
            Kn_tr = Kn(cv_index_n ~= ite_fold, :);
            Ku_tr = Ku(cv_index_u ~= ite_fold, :);
            Kp_te = Kp(cv_index_p == ite_fold, :);
            Kn_te = Kn(cv_index_n == ite_fold, :);
            Ku_te = Ku(cv_index_u == ite_fold, :);
            
            H.p = full(Kp_tr'*Kp_tr/size(Kp_tr, 1));
            H.n = full(Kn_tr'*Kn_tr/size(Kn_tr, 1));
            H.u = full(Ku_tr'*Ku_tr/size(Ku_tr, 1));
            
            m.p = full(mean(Kp_tr, 1));
            m.n = full(mean(Kn_tr, 1));
            m.u = full(mean(Ku_tr, 1));            
            
            clear Kp_tr Kn_tr Ku_tr
            
            for ite_lambda = 1:n_lambda
                lambda   = lambda_list(ite_lambda);
                for ite_eta = 1:n_eta
                    eta = eta_list(ite_eta);
                    if strcmp(ker_type, 'gauss')
                        LOG.trace(mfilename, ...
                            sprintf('fold: %d, sigma: %f, lambda: %f, eta: %.2f', ...
                            ite_fold, sigma, lambda, eta));
                    else
                        LOG.trace(mfilename, sprintf('fold: %d, lambda: %f, eta: %.2f', ...
                            ite_fold, lambda, eta));
                    end
                    theta_cv = solve(H, m, prior, lambda, eta);
                    gp = Kp_te*theta_cv;
                    gn = Kn_te*theta_cv;
                    gu = Ku_te*theta_cv;
                    
                    risk4eta = calc_eta_heu_unb(np, nn, prior, eta);
                    score_table(ite_sigma, ite_lambda, ite_eta) = ...
                        score_table(ite_sigma, ite_lambda, ite_eta) ...
                        + calc_loss(gp, gn, gu, prior, risk4eta)/n_fold;
                end % eta
            end % lambda
        end % hold
    end % sigma
end
[score_best, chosen_index] = min(score_table(:));    
[sigma_index, lambda_index, eta_index] = ind2sub(size(score_table), chosen_index);
sigma  = sigma_list(sigma_index);
lambda = lambda_list(lambda_index);
eta    = eta_list(eta_index);
LOG.trace(mfilename, sprintf('score: %f\n', score_best));
if strcmp(ker_type, 'gauss')
    LOG.trace(mfilename, sprintf('selected sigma=%.4f, lambda=%.4f\n', sigma, lambda));
else
    LOG.trace(mfilename, sprintf('selected lambda=%.4f\n', lambda));
end

[Kp, Kn, Ku] = calc_ker(dp, dn, du, sigma);
H.p = full(Kp'*Kp/np);
H.n = full(Kn'*Kn/nn);
H.u = full(Ku'*Ku/nu);

m.p = full(mean(Kp, 1));
m.n = full(mean(Kn, 1));
m.u = full(mean(Ku, 1));
theta = solve(H, m, prior, lambda, eta);
f_dec = make_func(theta, xc, sigma);
f_auc = @(xt_p, xt_n) mean(mean(bsxfun(@minus, f_dec(xt_p), f_dec(xt_n)') >= 0));


if nargout > 2
    outputs.sigma_index  = sigma_index;    
    outputs.lambda_index = lambda_index;    
    outputs.eta_index    = eta_index;    
    outputs.score_table  = score_table;    
end


end


function w = solve(H, m, prior, lambda, eta)

w = solve_gauss(H, m, prior, lambda, eta);

end


function w = solve_gauss(H, m, prior, lambda, eta)

b  = size(H.p, 2);

Reg = lambda*eye(b);
% Reg = lambda*speye(b);


Hpn = H.p - m.p'*m.n - m.n'*m.p + H.n;
hpn = (m.p - m.n)';

if eta >= 0
    Kpu = H.p - m.p'*m.u - m.u'*m.p + H.u;
    Kpp = 2*H.p - 2*m.p'*m.p;
    Hpu = (Kpu - prior*Kpp)/(1-prior);
    hpu = (m.p - m.u)'/(1-prior);
    
    w = ((1-eta)*Hpn + eta*Hpu + Reg)\((1-eta)*hpn + eta*hpu);
else
    eta = -eta;
    Knu = H.n - m.n'*m.u - m.u'*m.n + H.u;
    Knn = 2*H.n - 2*m.n'*m.n;
    Hnu = (Knu - (1-prior)*Knn)/prior;
    hnu = (m.u - m.n)'/prior;
    w = ((1-eta)*Hpn + eta*Hnu + Reg)\((1-eta)*hpn + eta*hnu);
end


end


function loss = calc_loss(gp, gn, gu, prior, eta)
% calculate the objective

loss_pn = mean(mean(zero_one_loss(bsxfun(@minus, gp, gn'))));

thp = prior;
thn = 1-prior;

np = length(gp);
nn = length(gn);

if eta >= 0
    lpu = mean(mean(zero_one_loss(bsxfun(@minus, gp, gu'))));
    lpp = sum(sum(zero_one_loss(bsxfun(@minus, gp, gp'))))/(np*(np-1));
    loss_pu = (lpu - thp*lpp + thp/(2*(np-1)))/thn;
    
    gamma = eta;
    loss  = (1-gamma)*loss_pn + gamma*loss_pu;
else
    lun = mean(mean(zero_one_loss(bsxfun(@minus, gu, gn'))));
    lnn = sum(sum(zero_one_loss(bsxfun(@minus, gn, gn'))))/(nn*(nn-1));
    loss_nu = (lun - thn*lnn + thn/(2*(nn-1)))/thp;
    
    gamma = -eta;
    loss  = (1-gamma)*loss_pn + gamma*loss_nu;
end

    function ret = zero_one_loss(m)
        ret = (1 - sign(m))/2;
        ret(m == 0) = 1/2;
    end
end


function func_dec = make_func(theta, xc, sigma)
global ker_type

switch ker_type
    case 'gauss'
        func_dec = @(x_test) exp(-calc_dist2(x_test, xc)/(2*sigma^2))*theta;
    case 'lm'
        func_dec = @(x_test) x_test*theta;
    case 'linear'
        func_dec = @(x_test) x_test*(xc'*theta);
end

end


function [Kp, Kn, Ku] = calc_ker(dp, dn, du, sigma)
global ker_type;

switch ker_type
    case 'gauss'
        Kp = exp(-dp/(2*sigma^2));
        Kn = exp(-dn/(2*sigma^2));
        Ku = exp(-du/(2*sigma^2));
    case {'lm', 'linear'}
        Kp = dp;
        Kn = dn;
        Ku = du;
end

end

function eta = calc_eta_heu_unb(np, nn, prior, eta)

tp = prior;
tn = 1-prior;

if eta >= 0
    a  = tn^2*(np-1);
    b  = tp^2*nn;
    eta = a/(a + b);
else
    a  = tn^2*np;
    b  = tp^2*(nn-1);
    eta = -b/(a + b);
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



