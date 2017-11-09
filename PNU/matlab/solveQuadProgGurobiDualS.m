% Solves a QP in the form of 
% 1/2 x'Hx  + f'x
% s.t.: Lx <= k
% With the Gurobi QP solver using simplex method
% INPUTS:
%	H: The matrix for the quadratic term
%	f: The matrix for the linear term
%	L: The linear constraint matrix
%	k: Linear constraint constant
%	vbasis, cbasis (OPTIONAL): basis vectors to attempt a warm start
% RETURNS:
%	x: the parameter vector
%	vbasis, cbasis: Vectors for hot start
function [x, vbasis, cbasis] = solveQuadProgGurobiDualS(H, f, L, k, lb, sense)
% function x = solveQuadProgGurobiDualS(H, f, L, k, lb)

% 	b = length(f);

	model.obj = f;
	model.Q = 1/2*H;

	model.A = L;
	model.rhs = k;   
	model.sense = sense;

	% This must be stated explicitly!
	% Default is a lower-bound of 0!
    model.lb = lb;

	% check the sizes
	model.modelsense = 'min';
%     model.modelsense = 'max';

% 	% attempt a hot start
	params.method = 1;
%     params.method = -1; % auto
	params.Presolve = 2;
	params.TimeLimit = 1200;
%     params.OptimalityTol = 1e-4;
	params.OutputFlag = 0;

    gurobiresult = gurobi(model, params);

	if (~strcmp(gurobiresult.status, 'OPTIMAL'))
		fprintf('Gurobi error [%s]\n', gurobiresult.status);
	end

	if (~isfield(gurobiresult, 'x'))
		keyboard;
	end

	x = gurobiresult.x;

	vbasis = gurobiresult.vbasis;
	cbasis = gurobiresult.cbasis;
end
