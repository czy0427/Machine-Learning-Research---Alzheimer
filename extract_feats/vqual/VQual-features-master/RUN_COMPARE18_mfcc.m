% ==========================================
%   Feature Extraction Demo for ComparE challenge
% ==========================================
% by Soo Jin Park (sj.park@ucla.edu)
% Feb. 2018
% ==========================================



%%
% =============================
%  Costomize here
% =============================
Fs = 16000;  % Files will be resampled to 8 kHz

%dirRoot = '/home/spapl/compare18/data/';
%dirFtr  = '/home/spapl/compare18/features/';

dirRoot = '/Users/suzinia/data/';
dirFtr = '/Users/suzinia/GoogleDrive/features/';

% dirPraat = 'C:/Suzinia/Dropbox/21Git/VQual/Toolkits/';
%dirMark = 'C:/Users/Soo Jin Park/Downloads/';
%pathLog = 'C:/Suzinia/Temp/log_vqual_extraction.txt';
%pathLog = ['/home/spapl/compare18/log/VQualExtraction' datestr(now,'mmddHHMM') '.txt'];

dirWork = cd;
addpath([dirWork '/Toolkits/VoiceBox'])
dirPraat = [dirWork '/Toolkits/'];



if isunix
    UNIX = true;
elseif ispc
    UNIX = false;
else
    error('OS not supported')
end

%{
if ~isdir(dirMark)
    mkdir(dirMark);
end
%}

% =============================
%  Get Sample data paths
% =============================



%fileDir = ['C:/Users/Soo Jin Park/Downloads/'];
%fileNames = {'test1.wav'};

% fileDir =  'C:/Suzinia/11 Data/SRE10_samples/';
% fileNames = {'female_badfile2_01.wav', 'female_badfile2_02.wav', 'female_badfile2_03.wav', 'female_badfile2_04.wav', 'female_badfile2_05.wav'};



%{
fileDir = 'C:/Suzinia/11 Data/2016_SID_VQual/29IndivSentences/FEMALE_SET100/';
dirPraat = 'C:/Suzinia/Dropbox/21Git/features/Toolkits/';

fileNames = {'031A_0_HS01_20',
    '033C_0_HS01_12',
    '037B_0_HS01_10',
    '038A_0_HS01_04',
    '038A_0_HS01_12',
    '038B_0_HS01_10',
    '048B_0_HS01_11',
    '049A_0_HS01_02',
    '049A_0_HS01_08',
    '052C_0_HS01_10',
    '060A_0_HS01_01',
    '061A_0_HS01_06',
    '072C_0_HS01_06',
    };
%}

%fileDir = 'D:/2016_SID_VQual_Data/29IndivSentences/FEMALE_SET91/'; fileNames = {'001A_0_HS01_04'};
% fileDir = 'D:/2016_SID_VQual_Data/50EntireVideo/FEMALE_SET100/';
% dirMark = 'D:/2016_SID_VQual_Mark/50EntireVideo/FEMALE_SET100/';
% dirPraat = 'D:/Soo/Dropbox/21Git/features/Toolkits/';



%TYPE = 'Crying';
%TYPE = 'SelfAssessedAffect';
TYPE = 'AtypicalAffect';
%TYPE = 'Heartbeat';

%SETID = 'FEMALE_SET100'; 


dirData = [dirRoot 'ComParE2018_' TYPE '/wav/' ];
%dirMark = ['/home/spapl/kaldi/egs/sre08/from_soo/SID_VQual_mark/' TYPE '/' SETID '/'];
%dirPraat = '/home/spapl/kaldi/egs/sre08/from_soo/git_repo/features/Toolkits/';

%numFrmFile = ['/home/spapl/kaldi/egs/sre08/from_soo/Results/ASV/' TYPE '#' SETID '#numFrames.txt'];

%dirOut = ['/home/spapl/compare18/features/' TYPE '/VQual/'];
dirOut = [dirFtr TYPE '/MFCC0/'];


if ~isdir(dirOut)
    mkdir(dirOut)
end

files = dir(dirData);
fileNames = {};
for i=1:length(files)
    currName = files(i).name;
    if length(currName)<3
        continue;
    end
    
    currName = strsplit(currName,'.');
    if length(currName)<2 || ~strcmp(currName{2} ,'wav')
        continue;
    end
    
    fileNames = cat(1, fileNames, currName{1});
end



% =============================
%  Process
% =============================
%h= waitbar(0, 'extracting features...');
parfor i = 1:length(fileNames)
    %waitbar( i/ length(fileNames));

    fileName = fileNames{i};
    %pathMark = [ dirMark fileName '.mark'];
    
    sndFilePath =  [dirData fileName '.wav']; channel =0;
    fprintf('Processing %s\n', fileName);

    %tic
    %[out, isValidFrame, CPP] = soo_featureExtraction(  sndFilePath, Fs, channel, dirPraat, pathLog, true, UNIX );
    %pathOut = [dirOut fileName '.vql'];
    %toc
    
    %----------------
    numCeps = 20;
    numFilt = [];
    windowSize = 25;
    frameShift = 10;
    
    [snd, origFs] = audioread(sndFilePath);
    if origFs ~= Fs
        snd = resample(snd, Fs, origFs);
        %snd = resample_soo( snd, Fs, origFs);
    end

    if isempty(numFilt)
        numFilt=floor(3*log(Fs));
    end
    [out,tc]=melcepst(snd,Fs,'MtdD0',numCeps,numFilt,floor((windowSize/1000)*Fs),floor((frameShift/1000)*Fs),0,0.5);
    pathOut = [dirOut fileName '.mfc'];
    %------------------
    
    % ::: Save features (each row corresponds a feature vector from one
    % frame)
    %pathOut = [dirOut fileName '.vql'];
    
    fid = fopen(pathOut , 'w');
    for j=1:size(out,1)
        for k=1:size(out,2)
            fprintf(fid, '%e\t', out(j,k));
        end
        
        fprintf(fid, '\n');
    end
    
    fclose(fid);
    
end

%close(h)