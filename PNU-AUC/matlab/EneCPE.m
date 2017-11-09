function [theta, dist] = EneCPE(xl, y, xu, alpha)
% EneCPE  Estimates the class prior p(y=+1) based on 
%   the energy distance minimization
% 
% Input:
%	xl: nl by d labeled sample matrix
%	y:  nl-dimensional label vector
% 	xu: nu by d unlabeled sample matrix
% 	alpha: The alpha parameter to use for the distance (Default 1)
% Output:
%	theta: estimated class prior p(y=+1)
%   dist: estimated energy distance between marginal distribution and
%     the model.
%
% Reference:
%   [1] H. Kawakubo, M.C. du Plessis, and M. Sugiyama
%     Computationally efficient class-prior estimation under class balance
%       change using energy distance.
%     IEICE Transactions on Information and Systems, vol.E99-D, no.1, 
%       pp.176-186, 2016. 
%
% (c) Hideko Kawakubo, Tokyo Institute of Technology, Japan.
%       kawakubo@ms.k.u-tokyo.ac.jp
%     Tomoya Sakai, The University of Tokyo, Japan.
%       sakai@ms.k.u-tokyo.ac.jp

narginchk(3, 4)
if (nargin<4)
    alpha = 1;
end

% get the classes
c = sort(unique(y));
nc = length(c);

% get the each labeled dataset
X = cell(size(c));
for i = 1:nc
    X{i} = xl(y==c(i), :);
end
Xte = xu;

% calculate b of the first term
b = zeros(nc, 1);
for i = 1:nc
    b(i, 1) = 2*mean(mean(AlphaDistance(X{i}, Xte, alpha)));
end

% calculate A of the second term
A = zeros(nc, nc);
for i = 1:nc
    for j = 1:nc
        A(i, j) = -mean(mean(AlphaDistance(X{i}, X{j}, alpha)));
    end
end

% calculate the third term
if nargout > 1
    T3 = mean(mean(AlphaDistance(Xte, Xte, alpha)));
end

% estimation of theta
As = A(1:(nc-1), 1:(nc-1));
a = A(1:(nc-1), nc);
one = ones((nc-1), 1);
Ac = A(nc, nc);
bs = b(1:(nc-1));
bc = b(nc);
%     Anew = As - one*a'- a*one' + Ac*(one*one');
Anew = bsxfun(@minus, bsxfun(@minus, As, a'), a) + Ac;
Anew = (Anew + Anew')/2;
bnew = 2*a - 2*Ac + bs - bc;

% calculate the solution
x0 = -bnew/(2*Anew);
x = min(max(x0, 0), 1); % p(y = -1)
theta = real(1-sum(x)); % p(y = +1)

if nargout > 1
    T1 = theta'*b;
    T2 = theta'*A*theta;    
    % calculate the total distance
    dist = T1 - (-T2) - T3;
end

end

% Calculates the |x-y|^\alpha distance between the row vectors in X and Y.
%   X: matrix of size n by d
%   Y: matrix of size m by d
%   dist: distance matrix of size n by m
function dist = AlphaDistance(X, Y, alpha)
    if alpha == 1
        dist = sqrt(bsxfun(@plus, sum(X.^2, 2), bsxfun(@minus, sum(Y.^2, 2)', 2*X*Y')));
    else
        dist = power(bsxfun(@plus, sum(X.^2, 2), bsxfun(@minus, sum(Y.^2, 2)', 2*X*Y')), alpha/2);
    end
    
end
