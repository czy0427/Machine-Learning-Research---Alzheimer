function CPP = func_GetCPP(y, Fs, F0, sampleshift, N_periods)
% CPP = func_GetCPP(y, Fs, F0, variables)
% Input:  y, Fs - from wavread
%         F0 - vector of fundamental frequencies
%         variables - global settings
% Output: CPP vector
% Notes:  Calculates the CPP according to the formula from Hillenbrand, 1994.
% Currently does not use textgrid information. For very very long files this 
% may be a process-bottleneck.
%
% Author: Yen-Liang Shue, Speech Processing and Auditory Perception Laboratory, UCLA
% Copyright UCLA SPAPL 2009
% 
% Modified by Soo Jin Park (sj.park@ucla.edu) 
% Copyrigut UCLA SPAPL 2016


lambda = 0; % linear regression regualarization factor

% N_periods = variables.Nperiods_EC;
% sampleshift = (Fs / 1000 * variables.frameshift);

N_ms = round(Fs / 1000); % Quefrency below 1ms are ignored as per Hillenbrand

CPP = zeros(length(F0), 1) * NaN;
for k=1:length(F0)
    ks = round(k * sampleshift);
    
    if ks <= 0 || ks > length(y) || isnan(F0(k))
        continue;
    end
    
    F0_curr = F0(k);
    N0_curr = Fs / F0_curr;
  
    ystart = round(ks - N_periods/2*N0_curr);
    yend   = round(ks + N_periods/2*N0_curr) - 1;
    
    if ystart<=0 || yend>length(y)
        continue;
    end;
    

  
    yseg = y(ystart:yend);  
    
    win = get_hamming(length(yseg));
    yseg = yseg .* win;
%     yseg = yseg .* hamming(length(yseg));  %windowing
    YSEG = fft(yseg);
    
    yseg_c = ifft(log(abs(YSEG)));
    yseg_c_db = 20*log10(abs(yseg_c));
  
    yseg_c_db = yseg_c_db(1:floor(length(yseg)/2));
    try
        [~, inx] = func_pickpeaks(yseg_c_db(N_ms:end)', 2*N0_curr);
    catch
        fprintf('failed to pick peaks: check if function exists \n');
        
        vals=[];
        inx = [];
    end
    [~, pinx] = min(abs(inx - N0_curr));
    
    if ~isempty(pinx)
        inx = inx(pinx(1));
        vals = yseg_c_db(inx+N_ms-1);
        
        x_train = [N_ms:length(yseg_c_db)].';
        y_train = yseg_c_db(N_ms:end);
        
        X = [ones( size(x_train,1), 1) x_train];
        % Y = y_train;
        
        Imat = eye(size(x_train,2)+1);
        Imat(1,1)=0;
        
        theta = (X.'*X + lambda*Imat)\X.' * y_train;
        
%         [theta]=func_linReg_fit(x_train, y_train, lambda);
        
        x_test = inx+N_ms-1;
        
        X_test = [ones( size(x_test,1), 1) x_test];
        base_val = X_test*theta;
%         base_val=func_linReg_pred(x_test, theta);
        
        CPP(k) = vals - base_val;
        
%     figure;
%     plot(yseg_c_db); hold on;
%     plot(inx+N_ms-1, vals, 'rx');
%     plot([N_ms:length(yseg_c_db)], polyval(p,[N_ms:length(yseg_c_db)]), 'g');
    
    end
    
end
end

function win = get_hamming(n)

% periodic window setting
n = n+1;

if ~rem(n,2)
    % Even length window
    half = n/2;
    
    x = (0:half-1)'/(n-1);
    w = 0.54 - 0.46*cos(2*pi*x);
    
    win = [w; w(end:-1:1)];
else
    % Odd length window
    half = (n+1)/2;

    x = (0:half-1)'/(n-1);
    w = 0.54 - 0.46*cos(2*pi*x);

    win = [w; w(end-1:-1:1)];
end

% periodic window setting
win(end) = [];

end

%{
function [theta]=func_linReg_fit(x_train, y_train, lambda)

X = [ones( size(x_train,1), 1) x_train];
% Y = y_train;

Imat = eye(size(x_train,2)+1);
Imat(1,1)=0;

theta = (X.'*X + lambda*Imat)\X.' * y_train;
end

function [y_hat]=func_linReg_pred(x_test, theta)

X_test = [ones( size(x_test,1), 1) x_test];
y_hat = X_test*theta;

end
%}
function [ymax,inx] = func_pickpeaks(y, winlen)
% [ymax,inx] = func_pickpeaks(y, winlen)
% Input:  y - from wavread
%         winlen - size of window to use
% Output: ymax - peak values
%         inx - peak positions
% Notes:  
%
% Author: Yen-Liang Shue and Markus Iseli, Speech Processing and Auditory Perception Laboratory, UCLA
% Copyright UCLA SPAPL 2009

% first, get all maxima
ymin = min(y);
y = y - ymin;


dy = [y(1); transpose(diff(y))];   % diff() gives a shift by 1: insert one sample
inx1 = diff(dy < 0);    % is +1 for maxima, -1 for minima


inx1(1) = 0; % do not accept maxima at begining
inx1(end) = 0; % do not accept maxima at end


inx = inx1 > 0;         % choose maxima only
ymax= y(inx);
inx = find(inx);
%plot(y);
%hold on;
%plot(inx,ymax,'or');
%hold off;

nofmax = length(ymax);
if nofmax==1
   return;
end
% now filter maxima with window of length winlen
for cnt = 1 : nofmax
   arr = inx(1:cnt);
   cmp = inx(cnt)-winlen;
   arr2 = arr>cmp;
   %ymax(arr2)
   [m, mi] = max(ymax(arr2));
   ymax(arr2)=-60000;
   ymax(mi+length(arr2)-sum(arr2))=m;
   %ymax(arr2)
end
temp = find(ymax>0);
inx  = inx(temp);
ymax = ymax(temp);
%plot(y);
%hold on;
%plot(inx,ymax,'or');
%hold off;
ymax = ymax + ymin;
end
