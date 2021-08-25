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
FTRSETNAME = 'VQual';     Fs = 8000;  
%FTRSETNAME = 'VQual0311'; Fs = 16000;  % Files will be resampled to 16 kHz


%TYPE = 'Crying';
TYPE = 'SelfAssessedAffect';

%TYPE = 'AtypicalAffect';
%TYPE = 'Heartbeat';

%%

dirRoot = '/home/spapl/compare18/data/';
dirFtr  = '/home/spapl/compare18/features/';
dirData = [dirRoot 'ComParE2018_' TYPE '/wav/' ];
pathLog = ['/home/spapl/compare18/log/VQualExtraction' datestr(now,'mmddHHMM') '.txt'];
%{
dirRoot = '/Users/suzinia/data/';
dirFtr = '/Users/suzinia/GoogleDrive/features/';
dirData = ['/Volumes/EtCetera/ComParE2018/data/ComParE2018_' TYPE '/wav/' ];
pathLog = ['/Users/suzinia/tmp/VQualExtraction' datestr(now,'mmddHHMM') '.txt'];
%}
%%

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

%{

<<<<<<< HEAD
=======
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
TYPE = 'SelfAssessedAffect';

%TYPE = 'AtypicalAffect';
%TYPE = 'Heartbeat';
>>>>>>> a5c2573c0559202c1da6168d6f498f99e140c04e

%}



%dirOut = ['/home/spapl/compare18/features/' TYPE '/VQual0305/'];
%dirOut = ['/home/spapl/compare18/features/' TYPE '/' FTRSETNAME '/'];
%dirOut = [dirFtr TYPE '/MFCCs/'];

dirOut = [dirFtr TYPE '/' FTRSETNAME '/'];


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

%fileNames = {'devel_0001'};


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

    try
        [output, isValidFrame, CPP] = soo_featureExtraction(  sndFilePath, Fs, channel, dirPraat, pathLog, true, UNIX );
    catch
        fprintf('an error occurred with %s', sndFilePath)
    end

    pathOut = [dirOut fileName '.vql'];

    %toc
    %{
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
    [output,tc]=melcepst(snd,Fs,'MtdD',numCeps,numFilt,floor((windowSize/1000)*Fs),floor((frameShift/1000)*Fs),0,0.5);
    pathOut = [dirOut fileName '.mfc'];
    %------------------
    %}
    
    % ::: Save features (each row corresponds a feature vector from one
    % frame)
    %pathOut = [dirOut fileName '.vql'];
    
    fid = fopen(pathOut , 'w');
    for j=1:size(output,1)
        for k=1:size(output,2)
            fprintf(fid, '%e\t', output(j,k));
        end
        
        fprintf(fid, '\n');
    end
    
    fclose(fid);
    
end

%close(h)