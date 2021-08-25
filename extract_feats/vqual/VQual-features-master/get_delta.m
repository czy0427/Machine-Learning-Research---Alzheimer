function dx=get_delta(x,D)

% the derivative is obtained
% as described in the HTK book (for HTK 3.4)

% inputs
%   x : [ FEATURE_DIMENSION FRM ] input matrix
%   D : delta window, 2*D+1 frames are involved


%----- initializing -----%
dx = NaN*ones(size(x));

%----- processing -----%
for nf = 1:size(x,2)
    try
        %----- calculate delta coefficients -----%
        numer = zeros(size(x,1),1);
        denom = 0;
        for theta = 1:D
            numer = numer + theta*(x(:, nf+theta)-x(:, nf-theta));
            denom = denom + 2*theta^2;
        end
        dx(:,nf) = numer/denom;
    catch
        %----- solve end-effect problem -----%
        if nf<=D
            dx(:,nf) = x(:,nf+1)-x(:,nf);
        elseif nf>=size(x,2)-D
            dx(:,nf) = x(:,nf)-x(:,nf-1);
        end
    end

end
end