function eta = calc_eta_heu1(np, nn, prior)

tp = prior;
tn = 1-prior;

As = 1/(np*nn);

if prior <= .5
    Bs = tp^2/(tn^2*np^2);    
    eta = As/(As + Bs);    
else
    Ds = tn^2/(tp^2*nn^2);
    eta = -As/(As + Ds);    
end


end

