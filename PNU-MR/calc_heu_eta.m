function eta = calc_heu_eta(np, nn, prior)
% CALC_HEU_ETA  Computes eta based on variance analysis with
%  \sigma_P(g)=\sigma_N(g)
% 
% See variance analysis for details.
% 
% (c) Tomoya Sakai, The University of Tokyo, Japan.
%     sakai@ms.k.u-tokyo.ac.jp

assert(np > 0 && nn > 0);
assert(0 < prior & prior < 1);

psi_p = prior^2/np;
psi_n = (1-prior)^2/nn;

eta = (psi_n - psi_p)/(psi_p + psi_n);

end