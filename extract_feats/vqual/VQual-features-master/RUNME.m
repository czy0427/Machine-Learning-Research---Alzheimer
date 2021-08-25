 % ==========================================
%   Feature Extraction Demo for ASV
% ==========================================
% by Soo Jin Park (sj.park@ucla.edu)
% Oct. 2016
%
% Last modified on
% Jul. 2017
%
%
% ==========================================



%%
% =============================
%  Costomize here
% =============================
Fs = 8000;  % Files will be resampled to 8 kHz

% dirPraat = 'C:/Suzinia/Dropbox/21Git/VQual/Toolkits/';
dirMark = 'C:/Users/Soo Jin Park/Downloads/';
pathLog = 'C:/Suzinia/Temp/log_vqual_extraction.txt';

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

if ~isdir(dirMark)
    mkdir(dirMark);
end

% =============================
%  Get Sample data paths
% =============================

%{
GENDER = 'f';
fileDir = '/home/spapl/kaldi/egs/sre08/from_soo/SID_VQual_data/29IndivSentences/FEMALE_SET100/';
fileName = '206C_0_HS08_08';

dirTmpFile = '/home/spapl/kaldi/egs/sre08/from_soo/git_repo/features/';
dirPraat   = '/home/spapl/kaldi/egs/sre08/from_soo/git_repo/features/Toolkits/';
%}

TYPE  = '29IndivSentences';
SETID = 'FEMALE_SET91'; 

% fileDir  = ['C:/Suzinia/11 Data/2016_SID_VQual_Data/' TYPE '/' SETID '/'];
% fileNames = {'001A_0_HS01_04.wav'};
% dirMark  = ['C:/Suzinia/11 Data/tmpSID_VQual_mark/' TYPE '/' SETID '/'];



% numFrmFile = ['C:/Suzinia/11 Data/' TYPE '#' SETID '#numFrames.txt'];


%{
fileDir = ['C:/Users/Soo Jin Park/Downloads/'];
% fileNames = {'tzzgp.sph'};
% fileNames = {'badfile1.wav','badfile2.wav', 'badfile3.wav'};
% fileNames = {'badfile2.wav'};
fileNames = {'testsph.sph'};
%}

fileDir = ['C:/Users/Soo Jin Park/Downloads/'];
fileNames = {'test1.wav'};

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

%{
% TYPE = '50EntireVideo';
% TYPE = '40EntirePrompts';
TYPE = '21ElongatedSentences';

SETID = 'FEMALE_SET100'; 


fileDir = ['/home/spapl/kaldi/egs/sre08/from_soo/SID_VQual_data/' TYPE '/' SETID '/'];
dirMark = ['/home/spapl/kaldi/egs/sre08/from_soo/SID_VQual_mark/' TYPE '/' SETID '/'];
dirPraat = '/home/spapl/kaldi/egs/sre08/from_soo/git_repo/features/Toolkits/';

numFrmFile = ['/home/spapl/kaldi/egs/sre08/from_soo/Results/ASV/' TYPE '#' SETID '#numFrames.txt'];



files = dir(fileDir);
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

%}


% =============================
%  Process
% =============================
for i = 1:length(fileNames)
    

    fileName = fileNames{i};
    pathMark = [ dirMark fileName '.mark'];
    
    sndFilePath =  [fileDir fileName]; channel =0;
    fprintf('Processing %s\n', fileName);

    tic
    [out, isValidFrame, CPP] = soo_featureExtraction(  sndFilePath, Fs, channel, dirPraat, pathLog, 0, UNIX );
    toc


end

