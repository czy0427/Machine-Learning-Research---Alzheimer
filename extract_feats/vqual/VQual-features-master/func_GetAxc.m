function [A1c, A2c, A3c] = func_GetAxc(Fs, F0, A1, A2, A3, F1, F2, F3, B1, B2, B3)
% [H1H2, H2H4] = func_GetH1H2_H2H4(H1, H2, H4, Fs, F0, F1, F2, B1, B2)
% Input:  H1, H2, H4, vectors
%         Fs - sampling frequency
%         F0 - vector of fundamental frequencies
%         Fx, Bx - vectors of formant frequencies and bandwidths
% Output: H1A1, H1A2, H1A3 vectors
% Notes:  Function produces the corrected versions of the parameters. They
% are stored as HxHx for compatibility reasons. Use func_buildMData.m to
% recreate the mat data with the proper variable names.
% Also note that the bandwidths from the formant trackers are not currently
% used due to the variability of those measurements.
%
% Author: Yen-Liang Shue, Speech Processing and Auditory Perception Laboratory, UCLA
% Copyright UCLA SPAPL 2009


if nargin == 8
%     B = func_getBWfromFMT([F1 F2 F3], F0, 'hm');
%     B1 = B(:,1);
%     B2 = B(:,2);
%     B3 = B(:,3);
  B1 = func_getBWfromFMT(F1, F0, 'hm');
  B2 = func_getBWfromFMT(F2, F0, 'hm');
  B3 = func_getBWfromFMT(F3, F0, 'hm');
end

% f = [F0 F0 2*F0 2*F0 4*F0 4*F0 F2K F2K F2K];
% Fx = [F1 F2 F1 F2 F1 F2 F1 F2 F3];
% Bx = [B1 B2 B1 B2 B1 B2 B1 B2 B3];

f  = [F1 F1 F2 F2 F3 F3 F3];
Fx = [F1 F2 F1 F2 F1 F2 F3];
Bx = [B1 B2 B1 B2 B1 B2 B3];


% Hxc = func_correct_iseli_z(f, , , Fs);


r = exp(-pi*Bx/Fs);
omega_x = 2*pi*Fx/Fs;
omega  = 2*pi*f/Fs;
a = r.^2 + 1 - 2*r.*cos(omega_x + omega);
b = r.^2 + 1 - 2*r.*cos(omega_x - omega);
% corr = -10*(log10(a)+log10(b));   % not normalized: H(z=0)~=0

num = r.^2 + 1 - 2*r.*cos(omega_x);   % sqrt of numerator, omega=0
Axc = -10*(log10(a)+log10(b)) + 20*log10(num);     % normalized: H(z=0)=1


A1c = A1 - Axc(:,1) - Axc(:,2);
A2c = A2 - Axc(:,3) - Axc(:,4);
A3c = A3 - Axc(:,5) - Axc(:,6) - Axc(:,7);




end



