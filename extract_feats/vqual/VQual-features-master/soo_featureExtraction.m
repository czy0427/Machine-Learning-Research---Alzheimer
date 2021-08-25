% ==========================================
%   Feature Extraction for ASV
% ==========================================
% by Soo Jin Park (sj.park@ucla.edu)
% Jul. 2017
%
% References
% Kreiman, J., Park, S. J., Keating, P. A., & Alwan, A. (2015). The Relationship Between Acoustic and Perceived Intraspeaker Variability in Voice Quality. In Interspeech (pp. 2357�2360).
% Huber, J., Stathopoulos, E. T., Curione, G. M., Ash, T. A., & Johnson, K. (1999). Formants in children, women, and men: the effect of vocal intensity variation. Journal of the Acoustical Society of America, 106(3, Pt.1), 1532�1542.
% ==========================================


function [out, isValidFrame, CPP] = soo_featureExtraction(  sndFilePath, Fs, channel, dirPraat, logFilePath, REMOVE_NAN, UNIX )


%%%%%%%%%%%
% Test Here
%%%%%%%%%%%
%fclose(fopen(strrep([sndFilePath,'.',num2str(channel),'.txt'],'/','#'),'w'));


%%
% ========================================
%   Configure feature set 
% ========================================
CORRECT_FMT     = false;    % correct formant effect on harmonics
NORMALIZE       = true;     % scale features to roughly fit them in [0 1] range

INCLUDE_FMT_AMP = true;     % include formant amplitudes A1,A2,A3
INCLUDE_F1_F2   = true;    % include the first and second formant frequencies
USE_ABSOLUTE_Hx = false;     % use Hx's instead of Hx-Hx's
USE_ABSOLUTE_Ax = true;    % use Ax's instead of H1-Ax's

USE_CQT_Hx = false;  % Use constant-Q transform to estimate Hxs

INCLUDE_SGR     = false;     % use subglottal resonance frequencies

INCLUDE_CPP     = true;  % include CPP in the feature matrix
                            % Even if it is set false, 
                            % the CPP as the output vector will be
                            % extracted anyway.
                            
INCLUDE_H5K     = true & Fs>10000;

%%
% ========================================
%	Parameter definitions
% ========================================
%D = 2;	% delta window
D = 2; 

windowsize = 25;      % [msec]
frameshift = 10;      % [msec]


%%
% ========================================
%              Read wav file
% ========================================
[~,fileName,extension] = fileparts(sndFilePath);
if strcmp(extension,'.wav')
    [snd,FsOrig] = audioread(sndFilePath);
else
    [snd,FsOrig,~,~,~] = readsph(sndFilePath);
end




snd = snd(:,channel+1);

snd = resample(snd,Fs,FsOrig);
% snd  = resample_soo(snd, Fs, FsOrig);  % simplified just for speed
% NOTE:
% It is recommended to use an advanced audio toolkit for the resampling
% if available (eg. sox)
% MATLAB 'resample' function is reported to result in some aliasing in 
% the high freqency region

tmpFileName = [fileName datestr(now,'HHMMSSFFF')]; 
tmpFilePath = [cd '/' tmpFileName '_' num2str(channel) '_temp.wav'];


%%
% ========================================
%	Save temporary wav file
% ========================================

audiowrite(tmpFilePath, .99*snd/max(abs(snd)), Fs);

sndFilePath = tmpFilePath;

TmpFileCreated = true;


%%
% ========================================
%  Start extracting features
% ========================================
data_len = floor(length(snd) / Fs * 1000 / frameshift);
sampleShift = (Fs / 1000 * frameshift);



%%
% -------------------------------
%	F0
% -------------------------------

%{
% Straight
[strF0, V] = func_StraightPitch_with_VAD(snd, Fs, VSparams, isVoiced);
strF0 = strF0(1:frameshift:end);  % drop samples if necessary

if (length(strF0) > data_len)
    strF0 = strF0(1:data_len);
elseif (length(strF0) < data_len)
    strF0 = [strF0; ones(data_len - length(strF0), 1) * NaN];
end
F0 = strF0;
%}

PraatParams.F0.F0max = 600;
PraatParams.F0.F0min = 75;

%{
switch gender
    % params defined in
    % Hautamaki et al. 2013
    case 'f'
        PraatParams.F0.F0max = 600;  % praat F0 settings
        PraatParams.F0.F0min = 100;
    case 'm'
        PraatParams.F0.F0max = 400;  % praat F0 settings
        PraatParams.F0.F0min =  75;
end
%}

PraatParams.F0.SilenceThreshold   = 0.03;
PraatParams.F0.VoicedThreshold    = 0.45;
PraatParams.F0.OctaveCost         = 0.01;
PraatParams.F0.OctaveJumpCost     = 0.35;
PraatParams.F0.VoicedUnvoicedCost = 0.14;
PraatParams.F0.KillOctaveJumps    = 0;
PraatParams.F0.Smooth             = 0;
PraatParams.F0.SmoothingBandwidth = 5;
PraatParams.F0.Interpolate        = 0;
PraatParams.F0.Method             = 'cc';

PraatParams.F0.FramePrecision = 1;

if Fs>10000
    PraatParams.Formants.MaxFormantFreq = 5000;
    PraatParams.Formants.NumFormants    = 4.5;

else
    PraatParams.Formants.MaxFormantFreq = 4000;
    PraatParams.Formants.NumFormants    = 3.5;    
end

PraatParams.Formants.MaxFormantFreq = 5000;
PraatParams.Formants.NumFormants    = 4.5;

% NOTE:
% When the samling freq is 8kHz, the max formant frequency should be less
% than 4kHz. 
% It is set to be 5kHz for now just to compare the performance to the
% previous code


Nperiods = 3;  % this sets out many pulses to use in the parameter estimation
Nperiods_EC = 5;  % both energy, CPP  and HNR calculations use this - larger window, more averaging


%%
% -------------------------------
%	Run Praat
% -------------------------------
if UNIX
    [isVoiced, pF0, pF1, pF2, pF3, ~, ~, ~, err] ...
        =  func_Praat_unix(sndFilePath, windowsize/1000, frameshift/1000, data_len, PraatParams, dirPraat);
else
    [isVoiced, pF0, pF1, pF2, pF3, ~, ~, ~, err] ...
        =  func_Praat_pc(sndFilePath, windowsize/1000, frameshift/1000, data_len, PraatParams, dirPraat);
end


if err
    logError('Praat failed', logFilePath, sndFilePath);    
    error('Praat failed');
else    
    F0 = pF0;
    F1 = pF1;   %B1 = pB1;
    F2 = pF2;   %B2 = pB2;
    F3 = pF3;   %B3 = pB3;
    
end

%% 
% -------------------------------
%   Choose F0 frames: Voiced + delta span
% -------------------------------
expandedVoiced = double(isVoiced);

% ::: Use a moving average filter 
a = 1;
b = 1/(1+2*D) * ones(1,(1+2*D));
try
    expandedVoiced  = filtfilt_soo(b,a,expandedVoiced);
catch
    fprintf('signal length: %d', length(expandedVoiced));
end
isExpVoiced  = expandedVoiced > 0;


F0(~isExpVoiced) = NaN;

%%
% -------------------------------
%	VQual feature set
% -------------------------------
try
    CPP = func_GetCPP(snd, Fs, F0, sampleShift, Nperiods_EC);
catch ME
    msg = sprintf('CPP estimation failed: %s', ME.message);
    logError(msg, logFilePath, sndFilePath);   
end

if USE_CQT_Hx
    try
        [H1, H2, H4, H2K, F2K] = func_GetHx_cqt(snd, Fs, F0, sampleShift, Nperiods,  PraatParams.F0.F0min);
    catch ME
        msg = sprintf('CQT-Hx estimation failed: %s', ME.message);
        logError(msg, logFilePath, sndFilePath);   
        
%         out = [];
        rethrow(ME)
%         return;
    end
else
    try
        if Fs>10000
            [H1, H2, H4, H2K, H5K, F2K, F5K] = func_GetHx(snd, Fs, F0, sampleShift, Nperiods);
        else
            [H1, H2, H4, H2K, ~, F2K, ~] = func_GetHx(snd, Fs, F0, sampleShift, Nperiods);
        end
    catch ME
        msg = sprintf('Hx estimation failed: %s', ME.message);
        logError(msg, logFilePath, sndFilePath); 
        rethrow(ME)
    end
end


if ~USE_ABSOLUTE_Hx
    if CORRECT_FMT
        try 
            [H1H2c, H2H4c, H42Kc] = func_GetHxHx(H1, H2, H4, H2K, Fs, F0, F2K, F1, F2, F3);
        catch ME
            msg = sprintf('Formant correction failed: %s', ME.message);
            logError(msg, logFilePath, sndFilePath);   
        end
        
        H1H2 = H1H2c;
        H2H4 = H2H4c;
        H42K = H42Kc;
    else
        H1H2u = H1-H2;
        H2H4u = H2-H4;
        H42Ku = H4-H2K;
        
        H1H2 = H1H2u;
        H2H4 = H2H4u;
        H42K = H42Ku;
        
        if INCLUDE_H5K
            H2KH5K = H2K-H5K;
        end
            
    end
end

if INCLUDE_FMT_AMP
    try
        [A1, A2, A3] = func_GetAx(snd, Fs, F0, F1, F2, F3, sampleShift, Nperiods);
    catch ME
        msg = sprintf('Ax estimation failed: %s', ME.message);
        logError(msg, logFilePath, sndFilePath); 
    end
end

if INCLUDE_SGR    
    [SG1, SG2, ~]=func_sgr(F1, F2, F3);
end

%%
% -------------------------------
%	Delete TMP_AUDIO wav file
% -------------------------------
if TmpFileCreated % delete the temporary file if it exists
    delete(sndFilePath);
end


%%
% -------------------------------
%	Concatenate features
% -------------------------------
if USE_ABSOLUTE_Hx
    %{
    ftr_cf  = [F0  F3  CPP  H1];

    % NOTE: The feature range is for female
    % No scaling factor available for Hx for this moment
    ftr_min = [ 93   2138   15   0];
    ftr_max = [275   3490   32   1];    
    
    %}
    ftr_cf  = [F0  F3    H1  H2  H4  H2K];

    % NOTE: The feature range is for female
    % No scaling factor available for Hx for this moment
    ftr_min = [ 93   2138      0   0   0   0];
    ftr_max = [275   3490      1   1   1   1];  
    
   
else
    ftr_cf =  [F0  F3    H1H2  H2H4  H42K];

    % NOTE: The feature range is for female
    % Values referred to Kreiman et al., 2015

    ftr_min = [ 93  2138  -2.64 -0.39  0.0];
    ftr_max = [275  3490  21.60 29.20 27.7];
end

if INCLUDE_H5K
    ftr_cf =  [ftr_cf H2KH5K];
    
    ftr_min = [ftr_min 0];
    ftr_max = [ftr_max 1];
end

if INCLUDE_CPP
    ftr_cf  = [ftr_cf CPP];
    ftr_min = [ftr_min 15];
    ftr_max = [ftr_max 32];
end

if INCLUDE_F1_F2
    ftr_cf  = [ftr_cf   F1    F2];
    ftr_min = [ftr_min 522   963];
    ftr_max = [ftr_max 1163 2701];
    
    L = size(ftr_cf,2);
    idxF1_F2 = [L-1 L];
end

if INCLUDE_FMT_AMP
    
    if USE_ABSOLUTE_Ax
        % Ax values referred to Huber et al., 1999
        ftr_cf  = [ftr_cf  A1   A2   A3];
        ftr_min = [ftr_min 66.0 62.0 52.0 ];
        ftr_max = [ftr_max 72.5 68.0 59.2 ];
    else
        ftr_cf  = [ftr_cf  H1-A1  H1-A2  H1-A3];
        ftr_min = [ftr_min -72.5  -68.0  -59.2];
        ftr_max = [ftr_max -66.0  -62.0  -52.0];
    end
end


if INCLUDE_SGR
    
    ftr_cf  = [ftr_cf  SG1 SG2];
    ftr_min = [ftr_min  400   1100 ];
    ftr_max = [ftr_max  800   1700 ];
    
end

%%
% -------------------------------
%	Feature scaling
% -------------------------------

if ~NORMALIZE
    ftr_min = zeros(size(ftr_min));
    ftr_max = ones(size(ftr_max));

    % ------------------------
    % Even if feature scaling is not enabled,
    % F0, F1-3 will be used after divided by 1000
    % ------------------------    
    ftr_max(1:2) = 1000; % for f0 and f3
    if INCLUDE_F1_F2
        ftr_max(idxF1_F2) = 1000;
    end
end



%{
if INCLUDE_FMT_AMP
    ftr_cf = [F0  F1 F2 F3  A1 A2 A3 CPP  H1H2   H2H4   H42K];
    
    if NORMALIZE
        % NOTE: The feature range is for female
        % Values referred to Kreiman et al., 2015
        % A1-A3 referred to Huber et al., 1999    
        ftr_min = [ 93  522  963 2138 66.0 62.0 52.0 15 -2.64 -0.39  0.0];
        ftr_max = [275 1163 2701 3490 72.5 68.0 59.2 32 21.60 29.20 27.7];
    else
        ftr_min = [   0    0    0    0  0  0  0  0  0  0  0];
        ftr_max = [1000 1000 1000 1000  1  1  1  1  1  1  1];
    end

else
    ftr_cf = [F0  F1 F2 F3  CPP  H1H2   H2H4   H42K];
    
    if NORMALIZE
        % NOTE: The feature range is for female
        % Values referred to Kreiman et al., 2015
        ftr_min = [ 93  522  963 2138 15 -2.64 -0.39  0.0];
        ftr_max = [275 1163 2701 3490 32 21.60 29.20 27.7];    
    else
        ftr_min = [   0    0    0   0   0  0  0  0];
        ftr_max = [1000 1000 1000 1000  1  1  1  1];  
        
%         ftr_min = [   0    0    0   0   0  0  0  0];
%         ftr_max = [1 1 1 1  1  1  1  1];  
    end
    
end
%}


ftr_rng = ftr_max - ftr_min;
ftr_norm =  (ftr_cf - repmat(ftr_min, [data_len 1]))./repmat(ftr_rng, [data_len 1]);

    
if sum(sum(isinf(ftr_norm))) > 0
    fid_log = fopen(logFilePath,'a');
    fprintf(fid_log, '%s ERROR: Infinity in ftr_norm. File name = %s\n', datestr(now),sndFilePath);
    fclose(fid_log);
end    

%%
% -------------------------------
%	Calculate deltas
% -------------------------------

% ::: First derivatives
ftrD1 = get_delta(ftr_norm.', D).';

% ::: Second derivatives
ftrD2 = get_delta(ftrD1.', D).';


if sum(sum(isinf(ftrD1))) > 0
    logError('Infinity in ftrD1', logFilePath, sndFilePath)
end

if sum(sum(isinf(ftrD2))) > 0
    logError('Infinity in ftrD2', logFilePath, sndFilePath)
end    
%%
% -------------------------------
%	Return output
% -------------------------------

ftrDeltas = [ftr_norm	ftrD1	ftrD2];

% ::: Select voiced frames only
nans = isnan(ftrDeltas);
nans = logical(sum(nans,2));

isValidFrame = ~nans & isVoiced;

%marks = get_mark(isValidFrame, frameshift/1000);
%f_mark = fopen(markFilePath,'w');
%for i = 1:size(marks,1)
%    fprintf(f_mark,'%s %.2f %.2f\n', sndFilePath, marks(i,1), marks(i,2));
%end
%fclose(f_mark);

if REMOVE_NAN
    out = ftrDeltas(isValidFrame, :);
    CPP = CPP(isValidFrame);
else
    out = ftrDeltas;
%     out(~isValidFrame,:) = zeros(size(out(~isValidFrame,:),1),size(out(~isValidFrame,:),2));
    out(~isValidFrame,:) = 0;
    
    CPP(~isValidFrame) = 0;
    
end




isValidFrame = double(isValidFrame);

%{
ftrDeltas = ftrDeltas( isVoiced, : );

nans = isnan(ftrDeltas);
nans = logical(sum(nans,2));
out = ftrDeltas( ~nans, :);
%}



end


function logError(message, logFilePath, sndFilePath)

    fid_log = fopen(logFilePath,'a');
    fprintf(fid_log, '%s ERROR: %s\n\t %s\n\n', datestr(now), message, sndFilePath );
    fclose(fid_log);

end