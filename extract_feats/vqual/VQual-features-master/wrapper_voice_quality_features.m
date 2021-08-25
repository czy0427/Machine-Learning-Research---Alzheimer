clear;clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('voicebox'); %DOWNLOAD VOICEBOX AND PROVIDE ITS PATH HERE
%%%% http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html

Fs = 44100;  % Files will be resampled to 8 kHz (CHANGE THIS IF YOU WANT A DIFFERENT SAMPLING RATE)
pathLog = 'Temp/log_vqual_extraction.txt';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dirWork = '/Users/vijayravi/Documents/UCLA/Research/SPAPL/dementia_2021/Alzheimer/extract_feats/vqual/VQual-features-master';  %location where the VQUal-features code is

addpath(dirWork);
addpath([dirWork '/Toolkits/VoiceBox']);
dirPraat = [dirWork '/Toolkits/'];

if isunix
    UNIX = true;
elseif ispc
    UNIX = false;
else
    error('OS not supported')
end
%%  Get Paths
feats_scp = 'train.scp';
fid = fopen(feats_scp);
M = textscan(fid, '%s%s', 'Delimiter',' ');
wav_files = M{1};
feat_files = M{2};
number_of_segments = length(wav_files);
if length(wav_files) ~= length(feat_files)
    disp("Error! Number of files mismatch");
    exit
end

%%
for cnt = 1:length(wav_files)
    
    fileName = wav_files{cnt};
    sndFilePath =  [fileDir fileName];
    fprintf('Processing %s\n', fileName);
    try
        [out, isValidFrame, CPP] = soo_featureExtraction(  sndFilePath, Fs, 0, dirPraat, pathLog, 0, UNIX );
        ftr = out(isValidFrame==1,:);
        s_ftr=sum(ftr,2)+0.000000000000001;
        ftr=ftr./repmat(s_ftr,1,size(ftr,2));
    catch
        ftr = [];
    end
    
   
    dlmwrite(feat_files{cnt},ftr);
    
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
