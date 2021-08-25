function [x,fval] = fminsearch_soo(funfcn,x,varargin)
%FMINSEARCH Multidimensional unconstrained nonlinear minimization (Nelder-Mead).
%   Reference: Jeffrey C. Lagarias, James A. Reeds, Margaret H. Wright,
%   Paul E. Wright, "Convergence Properties of the Nelder-Mead Simplex
%   Method in Low Dimensions", SIAM Journal of Optimization, 9(1):
%   p.112-147, 1998.
%
%   Copyright 1984-2012 The MathWorks, Inc.
%   $Revision: 1.1.6.3 $  $Date: 2012/08/21 00:30:36 $
%
% Optimized for VoiceSauce by Soo Jin Park


% n = numel(x);
n = 1;  % x is a scalar in this application
% numberOfVariables = n;


tolx = 1e-4;
tolf = 1e-4;

maxfun = 200;
maxiter = 200;


% Initialize parameters
rho = 1; chi = 2; psi = 0.5; sigma = 0.5;

onesn = 1;
two2np1 = 2;
one2n = 1;

% Set up a simplex near the initial guess.
% xin = x(:); % Force xin to be a column vector
% xin = x;
v = zeros(1,n+1); fv = zeros(1,n+1);
% v(:,1) = xin;    % Place input guess in the simplex! (credit L.Pfeffer at Stanford)
v(:,1) = x;    % Place input guess in the simplex! (credit L.Pfeffer at Stanford)
fv(:,1) = funfcn(x,varargin{:});
itercount = 0;




% Continue setting up the initial simplex.
% Following improvement suggested by L.Pfeffer at Stanford
usual_delta = 0.05;             % 5 percent deltas for non-zero terms
% zero_term_delta = 0.00025;      % Even smaller delta for zero elements of x

j=1;
% y = xin;
% y = x;
% if y ~= 0 % x will never be  0
%     y = (1 + usual_delta)*y;
    y = (1 + usual_delta)*x;
% else
%     y = zero_term_delta;
% end
v(:,j+1) = y;
% x(:) = y; f = funfcn(x,varargin{:});
f = funfcn(y,varargin{:});
fv(1,j+1) = f;


% sort so v(1,:) has the lowest function value
[fv,j] = sort(fv);
v = v(:,j);

% how = 'initial simplex';
itercount = itercount + 1;
func_evals = n+1;


% Main algorithm: iterate until
% (a) the maximum coordinate difference between the current best point and the
% other points in the simplex is less than or equal to TolX. Specifically,
% until max(||v2-v1||,||v2-v1||,...,||v(n+1)-v1||) <= TolX,
% where ||.|| is the infinity-norm, and v1 holds the
% vertex with the current lowest value; AND
% (b) the corresponding difference in function values is less than or equal
% to TolFun. (Cannot use OR instead of AND.)
% The iteration stops if the maximum number of iterations or function evaluations
% are exceeded
while func_evals < maxfun && itercount < maxiter
    if max(abs(fv(1)-fv(two2np1))) <= max(tolf,10*eps(fv(1))) && ...
            max(max(abs(v(:,two2np1)-v(:,onesn)))) <= max(tolx,10*eps(max(v(:,1))))
        break
    end
    
    % Compute the reflection point
    
    % xbar = average of the n (NOT n+1) best points
    xbar = sum(v(:,one2n), 2)/n;
    xr = (1 + rho)*xbar - rho*v(:,end);
    %x(:) = xr; fxr = funfcn(x,varargin{:});
    fxr = funfcn(xr,varargin{:});
    func_evals = func_evals+1;
    
    if fxr < fv(:,1)
        % Calculate the expansion point
        xe = (1 + rho*chi)*xbar - rho*chi*v(:,end);
%         x(:) = xe; fxe = funfcn(x,varargin{:});
        fxe = funfcn(xe,varargin{:});
        func_evals = func_evals+1;
        if fxe < fxr
            v(:,end) = xe;
            fv(:,end) = fxe;
            %             how = 'expand';
        else
            v(:,end) = xr;
            fv(:,end) = fxr;
            %             how = 'reflect';
        end
    else % fv(:,1) <= fxr
        if fxr < fv(:,n)
            v(:,end) = xr;
            fv(:,end) = fxr;
            %             how = 'reflect';
        else % fxr >= fv(:,n)
            % Perform contraction
            if fxr < fv(:,end)
                % Perform an outside contraction
                xc = (1 + psi*rho)*xbar - psi*rho*v(:,end);
%                 x(:) = xc; fxc = funfcn(x,varargin{:});
                fxc = funfcn(xc,varargin{:});
                func_evals = func_evals+1;
                
                if fxc <= fxr
                    v(:,end) = xc;
                    fv(:,end) = fxc;
                    how = 'contract outside';
                else
                    % perform a shrink
                    how = 'shrink';
                end
            else
                % Perform an inside contraction
                xcc = (1-psi)*xbar + psi*v(:,end);
%                 x(:) = xcc; fxcc = funfcn(x,varargin{:});
                fxcc = funfcn(xcc,varargin{:});
                func_evals = func_evals+1;
                
                if fxcc < fv(:,end)
                    v(:,end) = xcc;
                    fv(:,end) = fxcc;
                    how = 'contract inside';
                else
                    % perform a shrink
                    how = 'shrink';
                end
            end
            if strcmp(how,'shrink')
                for j=two2np1
                    v(:,j)=v(:,1)+sigma*(v(:,j) - v(:,1));
%                     x(:) = v(:,j); fv(:,j) = funfcn(x,varargin{:});
                    fv(:,j) = funfcn(v(:,j),varargin{:});
                end
                func_evals = func_evals + n;
            end
        end
    end
    [fv,j] = sort(fv);
    v = v(:,j);
    itercount = itercount + 1;
    
end   % while

x(:) = v(:,1);
fval = fv(:,1);




%{
%--------------------------------------------------------------------------
function f = checkfun(x,userfcn,varargin)
% CHECKFUN checks for complex or NaN results from userfcn.

f = userfcn(x,varargin{:});
% Note: we do not check for Inf as FMINSEARCH handles it naturally.
if isnan(f)
    error(message('MATLAB:fminsearch:checkfun:NaNFval', localChar( userfcn )));
elseif ~isreal(f)
    error(message('MATLAB:fminsearch:checkfun:ComplexFval', localChar( userfcn )));
end
%}
%{
%--------------------------------------------------------------------------
function strfcn = localChar(fcn)
% Convert the fcn to a string for printing

if ischar(fcn)
    strfcn = fcn;
elseif isa(fcn,'inline')
    strfcn = char(fcn);
elseif isa(fcn,'function_handle')
    strfcn = func2str(fcn);
else
    try
        strfcn = char(fcn);
    catch
        strfcn = getString(message('MATLAB:fminsearch:NameNotPrintable'));
    end
end
%}

