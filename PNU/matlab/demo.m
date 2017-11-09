function demo
global LOG
close all
rng(1);

% addpath(genpath('PATH_TO_MINFUNC_ROOT'));
% run('PATH_TO_GUROBI_SETUP/gurobi_setup');    
    
% make logger
log4m_make_instance('demo-pnu');
LOG.setCommandWindowLevel(LOG.INFO);

nl = 10;
nu = 200;
nt = 1000;
prior_l = .5;
prior_u = .3;

% options for PNU classifier
opts.model_type = 'lm';
opts.use_bias = true;
eta_list = -1:.1:1; % the candidates of \eta

% n_trials = 5;
n_trials = 20;
n_eta    = length(eta_list);

best_err = inf;
priors = zeros(n_trials, 1);
errs1  = zeros(n_trials, 1);
errs2  = zeros(n_trials, n_eta);
for ite_trial = 1:n_trials
    [x, y, xt_p, xt_n] = generate_data(nl, prior_l, nu, prior_u, nt);    
    priorh = EneCPE(x(y ~= 0, :), y(y ~= 0), x(y == 0, :)); % prior estimation
    np = sum(y ==  1);
    nn = sum(y == -1);
    opts.etab = calc_heu_eta(np, nn, priorh);
     [func, outs, funcs] = PNU_SL(x, y, priorh, eta_list, opts);    
%     [func, outs, funcs] = PNU_LL(x, y, priorh, eta_list, opts);    
%    [func, outs, funcs] = PNU_Ramp(x, y, priorh, eta_list, opts);    
    % computes misclassification rates 
    errs1(ite_trial) = 100*(prior_u*mean(func(xt_p) < 0) ...
            + (1-prior_u)*mean(func(xt_n) >= 0)); 
    % save the best decision boundary for illustration        
    if errs1(ite_trial) < best_err
        best_err = errs1(ite_trial);
        best_w = outs.w; 
    end
    % computes misclassification rates of classifiers obtained by 
    %   different \eta
    for ite_eta = 1:n_eta        
        errs2(ite_trial, ite_eta) = ...
            100*(prior_u*mean(funcs{ite_eta}(xt_p) < 0) ...
            + (1-prior_u)*mean(funcs{ite_eta}(xt_n) >= 0));
    end
    priors(ite_trial) = priorh;
end

fprintf('Mean of estimated priors: %.2f, True prior: %.2f\n', mean(priors), prior_u);
fprintf('Error: %.1f (%.1f)\n', mean(errs1), std(errs1)/sqrt(n_trials));

xp = x(y ==  1, :);
xn = x(y == -1, :);
xu = x(y ==  0, :);

%% Illustration of data points and estimated decision boundary
figure('Name', 'Demo');
hold on;
% plots data points
plot(xp(:, 1), xp(:, 2), 'bo', 'LineWidth', 1.8, 'MarkerSize', 10);
plot(xn(:, 1), xn(:, 2), 'rx', 'LineWidth', 1.8, 'MarkerSize', 10);
plot(xu(:, 1), xu(:, 2), 'k.', 'LineWidth', 1.8, 'MarkerSize', 10);

u1 = min(x(:, 1)); 
u2 = max(x(:, 1));

% plots optimal decision boundary
v1_opt = (log(prior_u/(1-prior_u))/2 - u1);
v2_opt = (log(prior_u/(1-prior_u))/2 - u2);
line([u1, u2], [v1_opt, v2_opt], 'LineWidth', 1.8, 'Color', 'k');

% plots estimated decision boundary
w = best_w(1:2);
intercept = best_w(3);
v1_est = (intercept - w(1)*u1)/w(2);
v2_est = (intercept - w(1)*u2)/w(2);
% v1_est = -w(1)*u1/w(2);
% v2_est = -w(1)*u2/w(2);
line([u1, u2], [v1_est, v2_est], 'LineWidth', 2.0, 'LineStyle', '-.');

xlabel('$x^{(1)}$', 'Interpreter', 'latex');
ylabel('$x^{(2)}$', 'Interpreter', 'latex');
xlim([-4, 4]);
ylim([-4, 4]);
title('Input data and estimated decision boundary', ...
    'Interpreter', 'latex');
legend('Positive samples', 'Negative samples', 'Unlabeled samples', ...
    'Optimal boundary', 'Estimated boundary', ...
    'Location', 'BestOutside');
set(gca, 'LineWidth', 0.8, 'FontSize', 10);
set(gcf, 'PaperUnits',    'centimeters');
set(gcf, 'PaperPosition', [0 0 12 6]);    
set(gcf, 'PaperType',     '<custom>');
set(gcf, 'PaperSize',     [12 6]);   

% print('-dpng', 'data-points.png');


%% Plots misclassification rates as a function of \eta
figure('Name', 'Error curve');
errorbar(eta_list, mean(errs2, 1), std(errs2, [], 1)/sqrt(n_trials), ...
    'LineWidth', 1.8);
xlabel('$\eta$', 'Interpreter', 'latex');
ylabel('Misclassification Rates (\%)', 'Interpreter', 'latex');
title(sprintf('Effect of $\\eta$ (%d trials)', n_trials), ...
    'Interpreter', 'latex');
xlim([-1.1, 1.1]);
set(gca, 'LineWidth', 0.8, 'FontSize', 10);
set(gcf, 'PaperUnits',    'centimeters');
set(gcf, 'PaperPosition', [0 0 8 6]);    
set(gcf, 'PaperType',     '<custom>');
set(gcf, 'PaperSize',     [8 6]);   

% print('-dpng', 'error-curve.png');


end

function [x, y, xt_p, xt_n] = generate_data(nl, prior_l, nu, prior_u, nt)
% nl:      the number of labeled samples
% prior_l: class prior for labeled data
% nu:      the number of unlabeled samples
% prior_u: class prior for unlabeled data
% nt:      the number of testing samples
%
% p(x | y=+1) = N( ( 1,..., 1_d)^T, , I_d)
% p(x | y=-1) = N( (-1,...,-1_d)^T, , I_d)

d    = 2;
mu_p = [ 1,  1];
mu_n = [-1, -1];

np   = sum(rand(nl, 1) < prior_l);
nn   = nl - np;
xp   = bsxfun(@plus, randn(np, d), mu_p);
xn   = bsxfun(@plus, randn(nn, d), mu_n);

nu_p = sum(rand(nu, 1) < prior_u);
nu_n = nu - nu_p;
xu_p = bsxfun(@plus, randn(nu_p, d), mu_p);
xu_n = bsxfun(@plus, randn(nu_n, d), mu_n);
xu   = [xu_p; xu_n];

x    = [xp; xn; xu];
y    = [ones(np, 1); -ones(nn, 1); zeros(nu, 1)];

xt_p = bsxfun(@plus, randn(nt, d), mu_p);
xt_n = bsxfun(@plus, randn(nt, d), mu_n);

end
