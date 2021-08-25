%% ===================================
%    SGR estimation
%
% Original code from Jinxi Guo, 2015
% Modified by Soo Jin Park (sj.park@ucla.edu), 2017
%
% Guo, Yang, Arsikere and Alwan (2017),
% "Robust speaker identification via fusion of subglottal resonance
% and cepstral features"
%
% SPAPL, UCLA
% ====================================

function [SG1, SG2, SG3]=func_sgr(F1, F2, F3)

% ::: Feature length
N = length(F1);

%% =====================
% SG1
% ======================
B31 = frq2bark(F3)-frq2bark(F1);

Eq1 = [0.011, -0.269, 1.322, 2.455]';

EstFeat = [B31.^3 B31.^2 B31 ones(N,1)];
B1s1    = EstFeat*Eq1;
SG1     = (bark2frq(frq2bark(F1)-B1s1));


%% =====================
% SG2
% ======================
B32 = frq2bark(F3)-frq2bark(F2);

Eq2 = [-0.004, 0.134, -1.958, 6.182]';

EstFeat = [B32.^3 B32.^2 B32 ones(N,1)];
B2s2 = EstFeat*Eq2;
SG2 = (bark2frq(frq2bark(F2)-B2s2));


%% =====================
% SG3
% ======================
Eq3 = [1.079, 763.676]';
SG3 = [SG2 ones(length(SG2),1)]*Eq3;



end