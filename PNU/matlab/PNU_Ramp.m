function [func_dec, outputs, funcs] = PNU_Ramp(x, y, prior, eta_list, options)
narginchk(5, 5);
global model_type LOG;

assert(0 < prior && prior < 1);
assert(isequal(unique(y), [-1; 0; 1]));
assert(-1 <= min(eta_list)  && max(eta_list) <= 1);

n_fold      = get_field_with_default(options, 'n_fold',      5);
model_type  = get_field_with_default(options, 'model_type', 'gauss');
lambda_list = get_field_with_default(options, 'lambda_list', logspace(-3, 1, 10));
etab        = get_field_with_default(options, 'etab',        []);

xp = x(y ==  1, :);
xn = x(y == -1, :);
xu = x(y ==  0, :);
    
np = sum(y ==  1);
nn = sum(y == -1);
nu = sum(y ==  0);

model_type = lower(model_type);
switch model_type
    case 'gauss'    
        LOG.info(mfilename, 'Gauss Kernel');
    case 'lm' % linear kernel, i.e., <x, x_\ell> for \ell = 1, \ldots, b
        LOG.info(mfilename, 'Linear Kernel');       
    otherwise
        LOG.error(mfilename, 'kernel type is invalid.');
        error('no kernel type\n');        
end
n_lambda = length(lambda_list);
n_eta    = length(eta_list);

cv_index_p = floor(((0:(np - 1))*n_fold)/np) + 1;
cv_index_p = cv_index_p(randperm(np));

cv_index_n = floor(((0:(nn - 1))*n_fold)/nn) + 1;
cv_index_n = cv_index_n(randperm(nn));

cv_index_u = floor(((0:(nu - 1))*n_fold)/nu) + 1;
cv_index_u = cv_index_u(randperm(nu));

% switch model_type
%     case 'gauss'
%         dp = calc_dist2(xp, xc);
%         dn = calc_dist2(xn, xc);
%         du = calc_dist2(xu, xc);        
%     case 'lm'
%         dp = [xp, ones(np, 1)];
%         dn = [xn, ones(nn, 1)];
%         du = [xu, ones(nu, 1)];
% end

% if strcmp(model_type, 'gauss')
%     sigma_list = get_field_with_default(options, 'sigma_list', ...
%         sqrt(median([dp(:); dn(:); du(:)]))*logspace(-2, 1, 10));    
% else
%     sigma_list = 1; % any value is ok
% end
sigma_list = logspace(-3, 0, 11);
n_sigma  = length(sigma_list);

score_table = zeros(n_sigma, n_lambda, n_eta);
if n_sigma == 1 && n_lambda == 1 && n_eta == 1
    score_table(1, 1, 1) = -inf;
    ite_list = [];
else
    ite_list = zeros(n_sigma, n_lambda*n_fold);
    for ite_sigma = 1:n_sigma
        sigma = sigma_list(ite_sigma);        
        for ite_fold = 1:n_fold;
            xp_tr = xp(cv_index_p ~= ite_fold, :);
            xp_te = xp(cv_index_p == ite_fold, :);
            xn_tr = xn(cv_index_n ~= ite_fold, :);
            xn_te = xn(cv_index_n == ite_fold, :);
            xu_tr = xu(cv_index_u ~= ite_fold, :);
            xu_te = xu(cv_index_u == ite_fold, :);
            
            for ite_eta = 1:n_eta
                eta = eta_list(ite_eta);
                [x_tr, y_tr, z_tr] = agg(xp_tr, xn_tr, xu_tr, eta);
                
                K_tr = calc_ker(x_tr, x_tr, sigma);                                
                K_tr = make_psd(K_tr);                
                for ite_lambda = 1:n_lambda
                    lambda = lambda_list(ite_lambda);
                    
                    if isempty(etab); mix = eta; else mix = etab; end
                    [x_te, y_te] = agg(xp_te, xn_te, xu_te, mix);
                    K_te = calc_ker(x_te, x_tr, sigma);                                                     
                    [~, ~, loss_cv, out_cccp] = cccp(K_tr, y_tr, z_tr, ...
                        K_te, y_te, prior, lambda, eta, mix);
                    ite_list(ite_sigma, ite_lambda) = out_cccp.T;
                    
                    score_table(ite_sigma, ite_lambda, ite_eta) = ...
                        score_table(ite_sigma, ite_lambda, ite_eta) + loss_cv/n_fold;
                end % lambda
            end % gamma
        end % fold
    end % sigma
end
score_list = zeros(n_eta, 1);
score_best = inf;
funcs      = cell(n_eta, 1);
w_list     = cell(n_eta, 1);
for ite_eta = 1:n_eta
    sub_score_table   = score_table(:, :, ite_eta);
    [~, chosen_index] = min(sub_score_table(:));    
    [sigma_index, lambda_index] = ind2sub(size(sub_score_table), chosen_index);
    score_list(ite_eta)       = sub_score_table(chosen_index);    
    if score_list(ite_eta) < score_best
        score_best  = score_list(ite_eta);
        best_sigma_index  = sigma_index;
        best_lambda_index = lambda_index;
        best_eta_index    = ite_eta;                
    end
    if nargout > 2
        sigma  = sigma_list(sigma_index);
        lambda = lambda_list(lambda_index);
        eta  = eta_list(ite_eta);
        [x, y, z] = agg(xp, xn, xu, eta);
        K = calc_ker(x, x, sigma);
        K = make_psd(K);
        if isempty(eta); mix = eta; else mix = eta; end
        w = cccp(K, y, z, [], [], prior, lambda, eta, mix);        
        w_list{ite_eta} = w;
        funcs{ite_eta} = make_func((z.*w)/lambda, x, sigma);   
    end
end
if nargout < 3 || n_eta == 1
    sigma  = sigma_list(best_sigma_index);
    lambda = lambda_list(best_lambda_index);
    eta  = eta_list(best_eta_index);
    [x, y, z] = agg(xp, xn, xu, eta);
    K = calc_ker(x, x, sigma);
    K = make_psd(K);
    if isempty(eta); mix = eta; else mix = eta; end
    w = cccp(K, y, z, [], [], prior, lambda, eta, mix);            
    func_dec = make_func((z.*w)/lambda, x, sigma);
else
    func_dec = funcs{best_eta_index};
    w = w_list{best_eta_index};
end

LOG.trace(mfilename, sprintf('score: %f\n', score_best));
if strcmp(model_type, 'gauss')
    LOG.trace(mfilename, sprintf('selected sigma=%.4f, lambda=%.4f\n', sigma, lambda));
else
    LOG.trace(mfilename, sprintf('selected lambda=%.4f\n', lambda));
end

if nargout > 1
    outputs.sigma_index  = best_sigma_index;    
    outputs.lambda_index = best_lambda_index;
    outputs.eta_index    = best_eta_index;
    outputs.score_table  = score_table; 
    outputs.score_list   = score_list;
    outputs.w            = w;
    
    outputs.ite_min = min([ite_list(:);    out_cccp.T]);
    outputs.ite_max = max([ite_list(:);    out_cccp.T]);
    outputs.ite_med = median([ite_list(:); out_cccp.T]);
    outputs.ite_avg = mean([ite_list(:);   out_cccp.T]);
end


end


function [alpha, u, loss, outputs] = cccp(K_tr, y_tr, z_tr, K_te, y_te, ...
    prior, lambda, gamma, mix)
% ConCave-Convex Procedure for combined risk with ramp loss
global LOG;

n = length(z_tr);
u = false(n, 1);

[f, L, H, k, lb, eta, sense] = make_quad_prog(K_tr, y_tr, z_tr, prior, lambda, u, gamma);

alpha     = solveQuadProgGurobiDualS(H, f, L, k, lb, sense);
g         = K_tr*(z_tr.*alpha)/lambda;
u         = z_tr.*g <= -1;

% prev_alpha     = alpha;
prev_obj       = -alpha'*H*alpha/2 - f'*alpha + sum(eta);

LOG.trace(mfilename, 'CCCP');
LOG.trace(mfilename, sprintf('\tT:%3d, ObjVal: %f, nSV: %3d, nU: %3d', ...
    0, prev_obj, sum(abs(alpha) > 1e-5), sum(u)));

% obj_plot = false;
% if obj_plot
%     objs = [];
%     objs = [objs, -alpha'*H*alpha/2 - f'*alpha + sum(eta)];    
%     tightens = [];
%     tightens = [tightens, -alpha'*H*alpha/2 - f'*alpha + sum(eta)];       
% end

T = 100;
for t = 1:T
    [f, L, H, k, lb, eta, sense] = make_quad_prog(K_tr, y_tr, z_tr, prior, lambda, u, gamma); % tightened bound
%     if obj_plot
%         tightens = [tightens, -alpha'*H*alpha/2 - f'*alpha + sum(eta)];       
%         tightens = [tightens, -alpha'*H*alpha/2 - f'*alpha + sum(eta)];           
%         objs = [objs, -alpha'*H*alpha/2 - f'*alpha + sum(eta)];
%     end
    
    alpha  = solveQuadProgGurobiDualS(H, f, L, k, lb, sense); % minimized
    obj    = -alpha'*H*alpha/2 - f'*alpha + sum(eta);
    
%     if prev_obj < obj && t > 1
%         alpha     = prev_alpha;
%         intercept = prev_intercept;
%         obj = prev_obj;
%         break;
%     end
    
%     if obj_plot
%         objs = [objs, obj];
%     end    
    LOG.trace(mfilename, sprintf('\tT:%3d, ObjVal: %f, nSV: %3d, nU: %3d', ...
        t, obj, sum(abs(alpha) > 1e-5), sum(u)));
    
%     prev_alpha     = alpha;
    prev_u         = u;       

    g = K_tr*(z_tr.*alpha)/lambda;
    u = z_tr.*g <= -1;
%     loss_list = [loss_list calc_loss(Kp, Kn, w, prior)];
%     if sum(abs(u - prev_u)) == 0 
    if sum(abs(u - prev_u)) == 0 || prev_obj < obj
        break;
    end       
    prev_obj       = obj;
end

if isempty(K_te)
    loss = obj;
else
    g_te = K_te*(z_tr.*alpha)/lambda;
    loss = calc_loss(g_te, y_te, prior, mix);
    LOG.trace(mfilename, sprintf('loss: %f', loss));
end

% if obj_plot
%     clf; hold on;
%     plot(1:length(objs), objs);
%     plot(1:length(tightens), tightens);
%     legend('minimized', 'tightened');
% end

% clf;
% hold on;
% stem(margin_list);
% line([1 length(margin_list)], [1 1]);

% plot(1:length(loss_list), loss_list);

outputs.T = t;

end


function [f, L, H, k, lb, eta, sense] = make_quad_prog(K, y, z, prior, lambda, u, eta)
% make vectors and matrixes for standard quadratic programming problem
% 
% minimize   1/2*alpha'*H*alpha + f'*alpha
% subject to L*alpha <= k

np = sum(y ==  1);
nn = sum(y == -1);
nu = sum(y ==  0);
n  = np + nn + nu;

if eta >= 0
    if np ~= 0; cp = (1 + eta)*prior/np;     else cp = 0; end
    if nn ~= 0; cn = (1 - prior)*(1-eta)/nn; else cn = 0; end
    if nu ~= 0; cu = eta/nu;                 else cu = 0; end
else
    eta = -eta;
    if np ~= 0; cp = (1 - eta)*prior/np;       else cp = 0; end
    if nn ~= 0; cn = (1 - prior)*(1 + eta)/nn; else cn = 0; end
    if nu ~= 0; cu = eta/nu;                   else cu = 0; end  
end

f   = -ones(n, 1);
c   = cp.*(y == 1) + cn.*(y == -1) + cu.*(y == 0);
eta = c.*u;
k   = c - eta;
L   =  speye(n);
H   =  sparse((z*z').*K/lambda);
lb  = -eta;
sense = repmat('<', 1, n);

end


function loss = calc_loss(g, y, prior, eta)
% calculate the loss of combined riks

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
    loss    = (1-eta)*loss_pn + eta*loss_pu;
else
    eta  = -eta;
    if nu ~= 0; fn_u = mean(g(y ==  0) <= 0); else fn_u = 0; end
    loss_nu = max(fn_u + (1 - prior)*fp - (1 - prior), 0) + (1 - prior)*fp;
    % loss_nu = fn_u + 2*(1 - prior)*fp - (1 - prior);
    loss    = (1-eta)*loss_pn + eta*loss_nu;
end

end


function func_dec = make_func(theta, xc, sigma)
global model_type

switch model_type
    case 'gauss'
        func_dec = @(x_test) exp(-calc_dist2(x_test, xc)/(2*sigma^2))*theta;
    case 'lm'
        func_dec = @(x_test) (x_test*xc')*theta;
end

end


function K = calc_ker(x, xc, sigma)
global model_type;

switch model_type
    case 'gauss'
        K = exp(-calc_dist2(x, xc)/(2*sigma^2));
    case 'lm'
        K = x*xc';
end

end


function K = make_psd(K)

min_eigval = min(eig(K));
if min_eigval <= 1.11e-5
    K = K + (abs(min_eigval) + 1.11e-5)*eye(size(K));
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



