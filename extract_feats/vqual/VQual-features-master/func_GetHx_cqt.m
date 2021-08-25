function [H1, H2, H4, H2K, F2K] = func_GetHx_cqt(y, Fs, F0,sampleshift, N_periods, F0min)
% [H1, H2, H4, isComplete] = func_GetH1_H2_H4(y, Fs, F0, handles, variables, textgridfile)
% Input:  y, Fs - from wavread
%         F0 - vector of fundamental frequencies
%         handles - from MessageBox, detects if user has pressed stop
%         variables - global settings
%         textgridfile - this is optional
% Output: H1, H2, H4 vectors
%         isComplete - flag to indicate if the process was allowed to
%         finish
% Notes:  Function is quite slow! Use textgrid segmentation to speed up the
% process.
%
% Author: Yen-Liang Shue, Speech Processing and Auditory Perception Laboratory, UCLA
% Copyright UCLA SPAPL 2009

% N_periods = variables.Nperiods;
% sampleshift = (Fs / 1000 * variables.frameshift);

H1  = zeros(length(F0), 1) * NaN;
H2  = zeros(length(F0), 1) * NaN;
H4  = zeros(length(F0), 1) * NaN;

H2K = zeros(length(F0), 1) * NaN;
F2K = zeros(length(F0), 1) * NaN;
% H5K = zeros(length(F0), 1) * NaN;
% F5K = zeros(length(F0), 1) * NaN; 

% isComplete = false;

Q    =  68;

df_low  = 0.1;     % search around f_est in steps of df_low (in Hz) 
df_high = 0.5;     % search around fxk_est in steps of df (in Hz)
fmin =  F0min * (1-df_low);
fmax =  2000  * (1+df_high);


N_max = ceil(Q *Fs / (fmin * (1-df_low)) );
H_max = floor(N_max/2);

L = length(y);

for k=1:length(F0)
    
    ks = round(k * sampleshift);
    if ks <= 0 || ks > L
        continue;
    end
    
    F0_curr = F0(k);
%     if F0min > F0_curr
%         fprintf('f0 = %.2f < F0min= %.2f\n',F0_curr, F0min);
%     end
    if isnan(F0_curr) || F0_curr == 0 || F0_curr<F0min
        % Skip if F0 is smaller than predefined F0min
        continue;
    end
    N0_curr = 1 / F0_curr * Fs;

    
    % ::: Process only in cases regular Hx is calculated
    ystart = round(ks - N_periods/2*N0_curr);
    yend = round(ks + N_periods/2*N0_curr) - 1;
    
    if ystart <= 0 || yend > L
        continue;
    end    
    

    
    

    
    
    % ::: Get the segment to process
    ystart = ks - H_max;
    yend   = ks + (N_max-H_max) - 1;
    


    
    yseg = zeros(N_max,1);    
    if ystart<=0
        tmp  = y( 1:yend);
        yseg(N_max - length(tmp) +1 :end) = tmp;        
    elseif yend>L
        tmp = y( ystart:L);
        yseg(1:length(tmp)) = tmp; 
    else
        yseg = y(ystart:yend);
    end
    %yseg = y(ystart:yend)';



    % -------
    % EstMaxVal
    % -------
    func = @(x)func_EstMaxVal_cqt(x, yseg, Fs, Q);
    
    % ------
    % Get harmonics
    % ------
    f0 = F0_curr;

    % ::: Process only when the high-freq component is calculated
    % NOTE: high-freq component use much shorter window and
    % if the widowed signal is more likely to be silence,
    % which results in NaN. Then the features in the frame will not be used
    % anyway.
    % (just to boost the speed) 
    % -----
    if func_GetSegEnergy(yseg, fmax, Fs, Q) == 0
        continue;
    end
    
    [h2k, f2k] = func_GetHx_high(func, 2000, f0);    

    
    
    nf0 = [f0; 2*f0; 4*f0];
    
    
    df_range = round(nf0*df_low); % search range (in Hz)
    
    nf_min = nf0 - df_range;
    nf_max = nf0 + df_range;
    
        
    [h] = func_GetHx_low(func, nf0,  nf_min, nf_max);

    

    

    
    H1(k)  = h(1);
    H2(k)  = h(2);
    H4(k)  = h(3);
    H2K(k) = h2k;
    F2K(k) = f2k;
    
    
%     if sum( isinf( [h; h2k] ) )
%         fprintf(' Err: infty in feature\n')
%     end

    %{
    if GET_H5K
        [h5k, f5k] = func_GetHx_high(func, 5000, f0);
        H5K(k) = h5k;
        F5K(k) = f5k;
    end
    %}
end



% isComplete = true;
end


function [E] = func_GetSegEnergy(data, f, Fs, Q)

N = floor(Fs*Q/f);

N1 = floor(N/2);
M = round(length(data)/2);
seg = data( M-N1+1: M-N1+N );

E =  sum( seg.^2);
end


function [h] = func_GetHx_low(func, nf0, nf_min, nf_max)
% find harmonic magnitudes in dB of time signal x
% around a frequency estimate f_est
% Fs, sampling rate
% x,  input row vector (is truncated to the first 25ms)
% df_range, optional, default +-5% of f_est


n = length(nf0);
h = NaN*ones(n,1);
% fh= NaN*ones(n,1);
for i=1:n
    [~, val] = fminsearchbnd_soo(func, 2*pi, nf_min(i), nf_max(i));
    h(i) = -1*val;
%     fh(i)= x;
end


end


function [h,fh]=func_GetHx_high(func,fxk_est,f0_est)
% find harmonic magnitudes in dB of time signal x
% around a frequency estimate f_est
% Fs, sampling rate
% x,  input row vector (is truncated to the first 25ms)
% df_range, optional, default +-5% of f_est

df = 0.5;     % search around fxk_est in steps of df (in Hz)
df_range = round(f0_est*df); % search range (in Hz)

f_min = fxk_est - df_range;
f_max = fxk_est + df_range;


[x, val] = fminsearchbnd_soo(func, 2*pi, f_min, f_max);  

h = -1 * val;
fh = x;
end

%{
function [f, X] = cqtransform(x, Fs, fmin, fmax, Q)
% CQT for one frame




f = [];
k = 0;
while 1
    
    f_k = ( 2^(1/24) )^k * fmin;
    
    if f_k > fmax
        break;
    else
        f = [f; f_k];
        k = k+1;
    end
end

K = length(f);

M = floor(length(x)/2);
X = NaN* ones(K,1);
for k=1:K
    N = floor(Fs*Q/f(k));
    

    
    N1 = floor(N/2);
    N2 = N-N1;
    seg = x( M-N1+1: M+N2 );
    if sum( seg.^2) == 0
        continue;
    end
    
    den = N;
    num = 0;    
%     w = ones(N,1); % rectangular window
    w = hann(N); % Hann window
%     w = flattopwin(N);
%     w = hamming(N);
    for n=1:N
%         num = num+ w(n)*seg(n)*exp( -1j*2*pi*Q*(n-1)/N);
        num = num+ w(n)*seg(n)*exp( -1j*2*pi*f(k)*(n-1)/Fs);
    end
    X(k) = num/den;

end

end
%}


function val = func_EstMaxVal_cqt(f, data, Fs, Q)
% f is the F0 estimate
% n = 0:length(data)-1;




%%%%%%%%%%%%%%

N = floor(Fs*Q/f);
% if N>length(data)
%     fprintf('data shorter than N=%d\n', N);
% end


N1 = floor(N/2);
M = round(length(data)/2);
seg = data( M-N1+1: M-N1+N );
% if sum( seg.^2) == 0
%     val = NaN;
%     return;
% end




%     w = ones(N,1); % rectangular window
%     w = hann(N); % Hann window
%     w = flattopwin(N);
% w0 = hamming(N);

w = hamming_window(N);


%{
den = N; num=0;
for n=1:N
    num = num+ w(n)*seg(n)*exp( -1j*2*pi*f*(n-1)/Fs);
end
z = num/den;
%}

%{
xx = 0:N-1;
e = exp( -1j * 2* pi * f/Fs *xx);
den = N; num=0;
for n=1:N
    num = num+ w(n)*seg(n)*e(n);
end
z = num/den;
%}

%{
xx = (0:N-1).';
e = exp(  -1j * 2* pi *f *xx /Fs);
z = sum(e.*w.*seg)/N;
%}

%{
xx = 0:N-1;
omega = -1i * 2* pi *f /Fs;
e = exp( xx * omega);

y = w.*seg;
z = (e*y)/N;
%}


% e = exp( -1j * 2* pi * f/Fs *(0:N-1) );

% e = exp( -1j * 2* pi * Q/N *(0:N-1) );


q = Q; n=N; i = 0;
while ~rem(n,2) && ~rem(q,2)
    q = q/2; n = n/2;
    i = i+1;
end
e  = exp( -1j * 2*pi * Q/N * (0:n-1));
for j = 1:i
    e = [e e];
end

    
z = e*(w.*seg)/N;


val = -20*log10(abs(z));
end


function w = hamming_window(n)
%SYM_WINDOW   Symmetric generalized cosine window.
%   SYM_WINDOW Returns an exactly symmetric N point generalized cosine 
%   window by evaluating the first half and then flipping the same samples
%   over the other half.

if ~rem(n,2)
    % Even length window
    half = n/2;
%     w = calc_window(half,n,window);
    x = (0:half-1)'/(n-1);
    w = 0.54 - 0.46*cos(2*pi*x);
    w = [w; w(end:-1:1)];
else
    % Odd length window
    half = (n+1)/2;
    %w = calc_window(half,n,window);
    x = (0:half-1)'/(n-1);
    w = 0.54 - 0.46*cos(2*pi*x);
    w = [w; w(end-1:-1:1)];
end

end


