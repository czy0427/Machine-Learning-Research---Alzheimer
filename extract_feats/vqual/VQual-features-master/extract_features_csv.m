% ==========================================
%   Feature Extraction for ASV
% ==========================================
% by Soo Jin Park (sj.park@ucla.edu)
% Jan. 2019
%
% References
% Kreiman, J., Park, S. J., Keating, P. A., & Alwan, A. (2015). The Relationship Between Acoustic and Perceived Intraspeaker Variability in Voice Quality. In Interspeech (pp. 2357–2360).
% Huber, J., Stathopoulos, E. T., Curione, G. M., Ash, T. A., & Johnson, K. (1999). Formants in children, women, and men: the effect of vocal intensity variation. Journal of the Acoustical Society of America, 106(3, Pt.1), 1532–1542.
% ==========================================
function extract_features_csv( sndFilePath, outFilePath, config, dirPraat)


% ========================================
%	Parameter definitions
% ========================================
D = 2;	% delta window

windowSize = 25;      % [msec]
frameShift = 10;      % [msec]

Nperiods    = 3;  % this sets out many pulses to use in the parameter estimation
Nperiods_EC = 5;  % both energy, CPP  and HNR calculations use this - larger window, more averaging


Fs     = config.desFs;
gender = config.gender;

% -------------------------------
%	Select Algorithm
% -------------------------------
alg_F0  = 'praat';
alg_FMT = 'praat';

% -------------------------------
%   Configure feature set 
% -------------------------------
% INCLUDE_FMT_AMP = config.INCLUDE_FMT_AMP;
% CORRECT_FMT     = config.CORRECT_FMT;



%%
% ========================================
%	Read Sound Data
% ========================================
[snd, origFs] = audioread(sndFilePath);
if origFs ~= Fs
    snd = resample(snd, Fs, origFs);
end

%%
% ========================================
%  Start extracting features
% ========================================

% -------------------------------
%	Create the output hdf5
% -------------------------------
if exist(outFilePath,'file')
    delete(outFilePath);
end

ftrLen = floor(length(snd) / Fs * 1000 / frameShift);
%{
if ~exist(outFilePath,'file')
    fid_w = H5F.create(outFilePath);
    H5F.close(fid_w);
    
    
    h5writeatt(outFilePath,'/','gender', gender);
    h5writeatt(outFilePath,'/','sndFilePath',sndFilePath);
    
    h5create(outFilePath,'/snd',[length(snd) 1]);
    h5write(outFilePath,'/snd', snd);
    h5writeatt(outFilePath,'/snd','Fs',Fs);
    
    
%     h5create(outFilePath, '/Fs', Fs);

    %ftrLen = floor(length(snd) / Fs * 1000 / frameShift);

    
    plist = 'H5P_DEFAULT';
    fid_w = H5F.open(outFilePath,'H5F_ACC_RDWR',plist);
    gid = H5G.create(fid_w, 'features', plist,plist,plist);
    H5G.close(gid);
    H5F.close(fid_w);
    
    
    
%     h5create(outFilePath,'/features',[ftrLen 100])

    % ------- default settings for the features ----------
    h5writeatt(outFilePath,'/features','ALG_F0', alg_F0);
    h5writeatt(outFilePath,'/features','ALG_FMT',alg_FMT);
    h5writeatt(outFilePath,'/features','frameShift[ms]',frameShift);
    h5writeatt(outFilePath,'/features','windowSize[ms]',windowSize);
    h5writeatt(outFilePath,'/features','featureLength', ftrLen);
    
    
end
%}



sampleShift = (Fs / 1000 * frameShift);

% -------------------------------
%	F0
% -------------------------------
switch lower(alg_F0)
    case 'straight'
        
        %TODO: if strF0 exists, use it
        
        % Straight
        [strF0, V] = func_StraightPitch_with_VAD(snd, Fs, VSparams, isVoiced);
        strF0 = strF0(1:frameShift:end);  % drop samples if necessary
        
        if (length(strF0) > ftrLen)
            strF0 = strF0(1:ftrLen);
        elseif (length(strF0) < ftrLen)
            strF0 = [strF0; ones(ftrLen - length(strF0), 1) * NaN];
        end
        F0 = strF0;
        
%         h5write(outFilePath,'strF0', strF0);
        
    case 'praat'
        % Praat
        PraatParams = get_praat_params(gender, Fs);
        
        
        % create temporary sound file
        [dataRoot,fileName,~] = fileparts(sndFilePath);
        tmpFilePath = [dataRoot '/tmp_' fileName datestr(now,'MMSS') '.wav'];
        audiowrite(tmpFilePath, snd, Fs);
        TMP_FILE_CREATED = true;
        
        [pVoiced, pF0, pF1, pF2, pF3, pF4, ~, ~, ~,~, err] ...
            =  func_Praat_wb(tmpFilePath, windowSize/1000, frameShift/1000, ftrLen, PraatParams, dirPraat);
        
        if err
            error('Praat error')
        else
            %{
            write_feature(outFilePath, 'pVoiced', double(pVoiced), ftrLen);
            write_feature(outFilePath, 'pF0', pF0, ftrLen);
            write_feature(outFilePath, 'pF1', pF1, ftrLen);
            write_feature(outFilePath, 'pF2', pF2, ftrLen);
            write_feature(outFilePath, 'pF3', pF3, ftrLen);
            
            
            if PraatParams.Formants.NumFormants  >= 4
                write_feature(outFilePath, 'pF4', pF4, ftrLen);
            end
            %}
            
            names =  {'pVoiced','pF0','pF1','pF2','pF3'};
            values = [double(pVoiced), pF0, pF1, pF2, pF3];
            
            % For extracting other features
%             isVoiced = pVoiced;
            F0 = pF0;
            F1 = pF1;   %B1 = pB1;
            F2 = pF2;   %B2 = pB2;
            F3 = pF3;   %B3 = pB3;
            
        end
        
        if TMP_FILE_CREATED
            delete(tmpFilePath);
        end

end


%% 
%{
% -------------------------------
%   Choose F0 frames: Voiced + delta span
% -------------------------------
expandedVoiced = double(isVoiced);

a = 1;
b = 1/(1+2*D) * ones(1,(1+2*D));
expandedVoiced  = filtfilt_soo(b,a,expandedVoiced);
isExpVoiced  = expandedVoiced>0;


F0(~isExpVoiced) = NaN;
%}


%%
% -------------------------------
% CPP
% -------------------------------
CPP = func_GetCPP(snd, Fs, F0, sampleShift, Nperiods_EC);
%write_feature(outFilePath, 'CPP', CPP, ftrLen);

names = cat(2,names, 'CPP');
values = cat(2, values, CPP);

%%
% -------------------------------
% Hx
% -------------------------------
if Fs > 10000
    GET_H5K = true;
else
    GET_H5K = false;
end

% [H1, H2, H4, H2K, F2K] = func_GetHx(snd, Fs, F0, sampleShift, Nperiods);

% 
[H1, H2, H4, H2K, F2K, H5K, F5K] = func_GetHx_wb(snd, Fs, F0, sampleShift, Nperiods, GET_H5K);

%{
write_feature(outFilePath, 'H1', H1, ftrLen);
write_feature(outFilePath, 'H2', H2, ftrLen);
write_feature(outFilePath, 'H4', H4, ftrLen);
write_feature(outFilePath, 'H2K', H2K, ftrLen);
write_feature(outFilePath, 'F2K', F2K, ftrLen);
%}

names = cat(2, names, {'H1', 'H2', 'H4', 'H2K', 'F2K', 'H5K', 'F5K'});
values = cat(2, values, [H1, H2, H4, H2K, F2K, H5K, F5K]);

%{
if GET_H5K
    write_feature(outFilePath, 'H5K', H5K, ftrLen);
    write_feature(outFilePath, 'F5K', F5K, ftrLen);
end
%}

%{
% Hxc ( only correction - no subtraction)
[H1c, H2c, H4c, H2Kc] = func_GetHxc(H1, H2, H4, H2K, Fs, F0, F2K, F1, F2, F3);

write_feature(outFilePath, 'H1c', H1c, ftrLen);
write_feature(outFilePath, 'H2c', H2c, ftrLen);
write_feature(outFilePath, 'H4c', H4c, ftrLen);
write_feature(outFilePath, 'H2Kc', H2Kc, ftrLen);
%}

%%
%{
% -------------------------------
% Hx*-Hx*
% -------------------------------
[H1H2c, H2H4c, H4H2Kc, H2KH5Kc] = func_GetHxHx_wb(H1, H2, H4, H2K, H5K, Fs, F0, F2K, F5K, F1, F2, F3);
write_feature(outFilePath, 'H1H2c', H1H2c, ftrLen);
write_feature(outFilePath, 'H2H4c', H2H4c, ftrLen);
write_feature(outFilePath, 'H4H2Kc', H4H2Kc, ftrLen);


if GET_H5K
    write_feature(outFilePath, 'H2KH5Kc', H2KH5Kc, ftrLen);
end

% -------------------------------
% Hx-Hx (no correction)
% -------------------------------
H1H2u   = H1-H2;
H2H4u   = H2-H4;
H4H2Ku  = H4-H2K;

write_feature(outFilePath, 'H1H2u', H1H2u, ftrLen);
write_feature(outFilePath, 'H2H4u', H2H4u, ftrLen);
write_feature(outFilePath, 'H4H2Ku', H4H2Ku, ftrLen);

if GET_H5K
    H2KH5Ku = H2K-H5K;
    
    write_feature(outFilePath, 'H2KH5Ku', H2KH5Ku, ftrLen);
end
%}

%%
%{
if CORRECT_FMT
    [H1H2c, H2H4c, H42Kc] = func_GetHxHx(H1, H2, H4, H2K, Fs, F0, F2K, F1, F2, F3);
    
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
end
%}
%%
%{
% Ax
[A1, A2, A3] = func_GetAx(snd, Fs, F0, F1, F2, F3, sampleShift, Nperiods);

write_feature(outFilePath, 'A1', A1, ftrLen);
write_feature(outFilePath, 'A2', A2, ftrLen);
write_feature(outFilePath, 'A3', A3, ftrLen);



% Hx-Ax --u 
H1A1u = H1-A1; 
H1A2u = H1-A2;
H1A3u = H1-A3;

write_feature(outFilePath, 'H1A1u', H1A1u, ftrLen);
write_feature(outFilePath, 'H1A2u', H1A2u, ftrLen);
write_feature(outFilePath, 'H1A3u', H1A3u, ftrLen);


% Hx-Ax --c
[A1c, A2c, A3c] = func_GetAxc(Fs, F0, A1, A2, A3, F1, F2, F3);
H1A1c = H1c-A1c;
H1A2c = H1c-A2c;
H1A3c = H1c-A3c;

write_feature(outFilePath, 'H1A1c', H1A1c, ftrLen);
write_feature(outFilePath, 'H1A2c', H1A2c, ftrLen);
write_feature(outFilePath, 'H1A3c', H1A3c, ftrLen);

%{
if INCLUDE_FMT_AMP
    [A1, A2, A3] = func_GetAx(snd, Fs, F0, F1, F2, F3, sampleShift, Nperiods);
    
    h5create(outFilePath,'/features/A1',[ftrLen 1]);
    h5create(outFilePath,'/features/A2',[ftrLen 1]);
    h5create(outFilePath,'/features/A3',[ftrLen 1]);
    
    h5write(outFilePath, '/features/A1',A1);    
    h5write(outFilePath, '/features/A2',A2);     
    h5write(outFilePath, '/features/A3',A3);
end
%}

% Energy
E = func_GetEnergy(snd, F0, Fs, Nperiods_EC, sampleShift);
write_feature(outFilePath, 'E', E, ftrLen);

% HNRs
[HNR05, HNR15, HNR25, HNR35] = func_GetHNR(snd, Fs, F0, Nperiods_EC, sampleShift);
write_feature(outFilePath, 'HNR05', HNR05, ftrLen);
write_feature(outFilePath, 'HNR15', HNR15, ftrLen);
write_feature(outFilePath, 'HNR25', HNR25, ftrLen);
write_feature(outFilePath, 'HNR35', HNR35, ftrLen);
%}
%%
%{
% -------------------------------
%  MFCC
% -------------------------------
numCeps = 20;
numFilt = [];

if isempty(numFilt)
    numFilt=floor(3*log(Fs));
end
[mfcc,tc]=melcepst(snd,Fs,'MtdD',numCeps,numFilt,floor((windowSize/1000)*Fs),floor((frameShift/1000)*Fs),0,0.5);

if (size(mfcc,1) > ftrLen)
    mfcc = mfcc(1:ftrLen, :);
else
    mfcc = [mfcc; ones(ftrLen - size(mfcc,1), size(mfcc,2)) *NaN];
end

for ncep = 1:numCeps
    write_feature(outFilePath, ['mfcc' num2str(ncep,'%02d')], mfcc(:,ncep), ftrLen);
end
%}

%{
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
if INCLUDE_FMT_AMP
    ftr_cf = [F0  F1 F2 F3  A1 A2 A3 CPP  H1H2   H2H4   H42K];
    
    % NOTE: The feature range is for female
    % Values referred to Kreiman et al., 2015
    % A1-A3 referred to Huber et al., 1999    
    ftr_min= [ 93  522  963 2138 66.0 62.0 52.0 15 -2.64 -0.39  0.0];
    ftr_max= [275 1163 2701 3490 72.5 68.0 59.2 32 21.60 29.20 27.7];

else
    ftr_cf = [F0  F1 F2 F3  CPP  H1H2   H2H4   H42K];
    
    % NOTE: The feature range is for female
    % Values referred to Kreiman et al., 2015
    ftr_min= [ 93  522  963 2138 15 -2.64 -0.39  0.0];
    ftr_max= [275 1163 2701 3490 32 21.60 29.20 27.7];
    
end
ftr_rng = ftr_max - ftr_min;

ftr_norm =  (ftr_cf - repmat(ftr_min, [data_len 1]))./repmat(ftr_rng, [data_len 1]);
%}
%{
%%
% -------------------------------
%	Calculate deltas
% -------------------------------

% ::: First derivatives
ftrD1 = get_delta(ftr_norm.', D).';

% ::: Second derivatives
ftrD2 = get_delta(ftr_norm.', D).';

%%
% -------------------------------
%	Return output
% -------------------------------

ftrDeltas = [ftr_norm	ftrD1	ftrD2];

% ::: Select voiced frames only
ftrDeltas = ftrDeltas( isVoiced, : );

nans = isnan(ftrDeltas);
nans = logical(sum(nans,2));
out = ftrDeltas( ~nans, :);
%}

write_csv(outFilePath, names, values, frameShift);


end

function write_csv(outFilePath, names, values, frameShift)

fid=fopen(outFilePath, 'w');

fprintf(fid, ' t');
for j=1:size(names,2)
    fprintf(fid, ';%s', names{j});
end
fprintf(fid, '\n');

for i=1:size(values,1)
    fprintf(fid, 'VoiceSauce;%.3f',i*frameShift/1000);
    for j=1:size(values,2)
        fprintf(fid, ';%.6f', values(i,j));
    end
    fprintf(fid, '\n');
end

fclose(fid);


end


function PraatParams = get_praat_params(gender, Fs)

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

%PraatParams.Formants.MaxFormantFreq = 5000;

if Fs == 8000
    PraatParams.Formants.MaxFormantFreq = 4000;
    PraatParams.Formants.NumFormants    = 3.5;
elseif Fs == 16000
    PraatParams.Formants.MaxFormantFreq = 5000;
    PraatParams.Formants.NumFormants    = 4.5;
else
    error('undefined Fs')
end


end

