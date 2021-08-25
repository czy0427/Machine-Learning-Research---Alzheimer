function [x,fval] = fminsearchbnd_soo(fun,x0u, LB,UB)
% Author: John D'Errico
% E-mail: woodchips@rochester.rr.com
% Release: 4
% Release date: 7/23/06
%
% Optimized for VoiceSauce by Soo Jin Park
% sj.park@ucla.edu
%


% stuff into a struct to pass around
% params.args = varargin;
params.LB = LB;
params.UB = UB;
params.fun = fun;


% now we can call fminsearch, but with our own
% intra-objective function.
[xu,fval] = fminsearch_soo(@intrafun,x0u,params);

% undo the variable transformations into the original space
% x = xtransform(xu,LB, UB);

% lower and upper bounds
x = ((sin(xu)+1)/2) *(UB - LB) + LB;
% just in case of any floating point problems
x = max(LB,min(UB,x));

end % mainline end

% ======================================
% ========= begin subfunctions =========
% ======================================
function fval = intrafun(x,params)
% transform variables, then call original function

% transform
% xtrans = xtransform(x,params.LB, params.UB);

% lower and upper bounds
xtrans = ((sin(x)+1)/2) *(params.UB - params.LB) + params.LB;
% just in case of any floating point problems
xtrans = max(params.LB,min(params.UB,xtrans));

% and call fun
fval = feval(params.fun, xtrans);
end % sub function intrafun end

%{
% ======================================
function xtrans = xtransform(x,LB, UB)
% converts unconstrained variables into their original domains

% lower and upper bounds
xtrans = ((sin(x)+1)/2) *(UB - LB) + LB;
% just in case of any floating point problems
xtrans = max(LB,min(UB,xtrans));


end % sub function xtransform end
%}




