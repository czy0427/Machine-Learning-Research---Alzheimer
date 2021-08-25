function [H1, H2, H4, H2K, H5K, F2K, F5K] = func_GetHx(y, Fs, F0, sampleshift, N_periods)
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
H5K = zeros(length(F0), 1) * NaN;
F2K = zeros(length(F0), 1) * NaN;
F5K = zeros(length(F0), 1) *NaN;

% isComplete = false;



for k=1:length(F0)
    

    
    ks = round(k * sampleshift);
    if ks <= 0 || ks > length(y)
        continue;
    end
    
    F0_curr = F0(k);
    if isnan(F0_curr) || F0_curr == 0
        continue;
    end
    N0_curr = 1 / F0_curr * Fs;
    
    
    ystart = round(ks - N_periods/2*N0_curr);
    yend = round(ks + N_periods/2*N0_curr) - 1;
    
    if ystart <= 0 || yend > length(y)
        continue;
    end
    
    
    yseg = y(ystart:yend)';
    
    if sum( yseg.^2 ) == 0 
        % Skip the segments with all-zero samples
        continue;
    end
    
%     if ismember(k, [6393, 6394, 9210,9211, 21416, 21417, 28554, 28555])
%         fprintf('frm %d yseg sum: %.2f\n',k, sum( yseg.^2 ));
%     end

    % -------
    % EstMaxVal
    % -------
    func = @(x)func_EstMaxVal(x, yseg, Fs);
    
    % ------
    % Get harmonics
    % ------
    f0 = F0_curr;
    
    
    nf0 = [f0; 2*f0; 4*f0];
    
    df = 0.1;     % search around f_est in steps of df (in Hz) 
    df_range = round(nf0*df); % search range (in Hz)
    
    nf_min = nf0 - df_range;
    nf_max = nf0 + df_range;
    
        
    [h] = func_GetHx_low(func, nf0,  nf_min, nf_max);

    
    % -----
    [h2k, f2k] = func_GetHx_high(func, 2000, f0);
    if Fs>10000
        [h5k, f5k] = func_GetHx_high(func, 5000, f0);
        H5K(k) = h5k;
        F5K(k) = f5k;
    end
    
    H1(k)  = h(1);
    H2(k)  = h(2);
    H4(k)  = h(3);
    H2K(k) = h2k;
    F2K(k) = f2k;
end



% isComplete = true;
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


function val = func_EstMaxVal(x, data, Fs)
% x is the F0 estimate
n = 0:length(data)-1;
v = exp(-1i*2*pi*x*n/Fs);
val = -1 * 20*log10(abs(data * v'));
end





%{

% this function is used from 1/8/09 onwards - optimization used
%--------------------------------------------------------------------------
function [h,fh]=func_GetHarmonics(data,f_est,Fs)
% find harmonic magnitudes in dB of time signal x
% around a frequency estimate f_est
% Fs, sampling rate
% x,  input row vector (is truncated to the first 25ms)
% df_range, optional, default +-5% of f_est

df = 0.1;     % search around f_est in steps of df (in Hz)
df_range = round(f_est*df); % search range (in Hz)

f_min = f_est - df_range;
f_max = f_est + df_range;

f = @(x)func_EstMaxVal(x, data, Fs);
options = optimset('Display', 'off');
warning off;
%[x, val, exitflag, output] = fmincon(f, f_est, [], [], [], [], f_min, f_max, [], options);


[x, val, exitflag, output] = fminsearchbnd(f, f_est, f_min, f_max, options);  

h = -1 * val;
fh = x;
end
%}



% This function was used up to 1/8/09
% %--------------------------------------------------------------------------
% function [h,f]=func_GetHarmonics_old(x,f_est,Fs,df_range)
% % find harmonic magnitudes in dB of time signal x
% % around a frequency estimate f_est
% % Fs, sampling rate
% % x,  input row vector (is truncated to the first 25ms)
% % df_range, optional, default +-5% of f_est
% 
% df = 0.1;     % search around f_est in steps of df (in Hz)
% if nargin<4
%     df_range = round(f_est*0.1); % search range (in Hz)
% end
% f = f_est-df_range:df:f_est+df_range;
% f = f(:);     % make column vector
% 
% %x = x(1:round(Fs*0.025));  % analyze 25ms
% %x = hamming(length(x)).*x;
% n = 0:length(x)-1;
% v = exp(-i*2*pi*f*n/Fs);
% h = 20*log10(abs(x' * v.' ));
% %figure; plot(f,h);
% [h,inx]=max(h);
% f=f(inx);
