function demo
close all; clc;
global LOG

log4m_make_instance(['PNU-AUC-' date]);
LOG.setCommandWindowLevel(LOG.INFO);

prior = .1;
np = 10;
nn = 90;
nu = 1000;

n_trials = 5;
auc_pnu_mr_list  = zeros(n_trials, 1);
auc_pnu_auc_list = zeros(n_trials, 1);

for ite_trial = 1:n_trials
    rng(ite_trial);
     [xp, xn, xu, xt, yt] = load_ringnorm(np, nn, nu, prior);
    
    xl = [xp; xn];
    yl = [ones(np, 1); -ones(nn, 1)];
    x  = [xl; xu];
    y  = [yl; zeros(nu, 1)];
    xt_p = xt(yt == +1, :);
    xt_n = xt(yt == -1, :);
    
    opts = [];
    n     = np + nn + nu;    
    dist2 = calc_dist2(x(randperm(n, min(n, 500)), :), ...
        x(randperm(n, min(n, 500)), :));
    opts.sigma_list = sqrt(median(dist2(:)))*[1/8, 1/4, 1/2, 1, 1.5, 2];
    eta_list = -.9:.1:.9;
    [f_mr, ~, ~] = PNU_SL(x, y, prior, eta_list, opts);    
    [f_pnu, ~, ~] = PNU_AUC_SL(x, y, prior, eta_list, opts);
    
    auc_pnu_mr_list(ite_trial)  = calc_auc(f_mr, xt_p, xt_n);
    auc_pnu_auc_list(ite_trial) = calc_auc(f_pnu, xt_p, xt_n);
end

fprintf('\n');
fprintf('AUC\n');
fprintf('%.3f (%.3f) [PNU]\n', ...
    mean(auc_pnu_mr_list), std(auc_pnu_mr_list)/sqrt(n_trials));
fprintf('%.3f (%.3f) [PNU-AUC]\n', ...
    mean(auc_pnu_auc_list), std(auc_pnu_auc_list)/sqrt(n_trials));

figure('Name', 'ROC');
subplot(1, 2, 1);
[fp_mr_list, tp_mr_list] = perfcurve(yt, f_mr(xt), +1);
plot(fp_mr_list, tp_mr_list)
title('PNU');

subplot(1, 2, 2);
[fp_pnu_list, tp_pnu_list] = perfcurve(yt, f_pnu(xt), +1);
plot(fp_pnu_list, tp_pnu_list)
title('PNU-AUC')

end


function auc = calc_auc(f, xp, xn)

gp = f(xp);
gn = f(xn);

auc = mean(mean(bsxfun(@ge, gp, gn')));

end

function dist2 = calc_dist2(x, xc)
% make n by b squared-distance matrix, 
%   n is the number of samples, b is the number of basis functions.

dist2 = bsxfun(@plus, sum(x.^2, 2), bsxfun(@minus, sum(xc.^2, 2)', 2*x*xc'));

end




