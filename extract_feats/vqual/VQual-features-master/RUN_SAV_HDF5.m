% ================================================================
%   Feature Extraction to HDF5 Files
% ================================================================
% Extract features and save them into *.hdf5 files
%
% by Soo Jin Park (sj.park@ucla.edu)
% Copyright Nov. 2016
% ================================================================

%%
% ================================================================
%  Set up parameters
% ================================================================


IS_WB        = true;    % if true, Fs=16000, else, Fs=8000
CLEANUP_HDF5 = false;    % Clean up *.hdf5

DELTA_WINDOW = 2;

%{
% ::: Feature set configuration
% config.INCLUDE_FMT_AMP =  true;    % include formant amplitude? (Boolean)
% config.CORRECT_FMT     =  true;    % correct formant effect when estimating Hx-Hx? (Boolean)
% -------------------------------------------------------------------
% Experiment 1: INCLUDE_FMT_AMP = false, CORRECT_FMT = true (done)
% Experiment 2: INCLUDE_FMT_AMP = true,  CORRECT_FMT = true
% Experiment 3: INCLUDE_FMT_AMP = false, CORRECT_FMT = false
% Experiment 4: INCLUDE_FMT_AMP = true,  CORRECT_FMT = false 
% -------------------------------------------------------------------
%}




% ::: Sub-Directory with data 
dirData = {
    ['29IndivSentences/FEMALE_SET100/'];
    ['29IndivSentences/MALE_SET100/']};

% ---------------------------------------------
if IS_WB
    config.desFs = 16000;
    savFolder = 'wb_16k/';
else
    config.desFs  = 8000;
    savFolder = 'nb_8k/';
end

% ---------------------------------------------

% ---------------------------------------------
cmpName = getenv('computername');
switch cmpName
    case 'MACAIR-WIN'
        dataRoot = 'C:/Suzinia/11 Data/2016_SID_VQual/';
        saveRoot = 'C:/Suzinia/11 Data/2016_SID_hdf5/';
        dirPraat = 'C:/Suzinia/Dropbox/21Git/speaker-recognition/VQual_Feature_Extraction/Toolkits/';

    case ''
        dataRoot = '/home/spapl/kaldi/egs/sre08/from_soo/SID_VQual_data/';
        saveRoot = '/home/spapl/kaldi/egs/sre08/from_soo/SID_VQual_data/';
        dirPraat = '/home/spapl/kaldi/egs/sre08/from_soo/git_repo/speaker-recognition/VQual_Feature_Extraction/Toolkits/';

    otherwise
        error('undefined computer')
end


%%
% ================================================================
%  Process
% ================================================================

% ---------------------------------------------
%  Count the number of files
% ---------------------------------------------
NumFiles = 0;
for n = 1:length(dirData)
    currDirData = [dataRoot dirData{n}];
    
    dataList = dir(currDirData);
    NumFiles = NumFiles + length(dataList);

end

% ---------------------------------------------
%  Extract features
% ---------------------------------------------
cnt = 0;
h = waitbar(0, 'Extracting features...');
for n = 1:length(dirData)        
    currDirData = [dataRoot dirData{n}];
    currDirSave = [saveRoot dirData{n}];
    
    % -------------------------------
    %	Get Gender
    % -------------------------------
    tmpSpl = strsplit(currDirData,'/');
    gender = lower(tmpSpl{end-1}(1));
    if gender ~= 'f' && gender ~= 'm'
        error('invalid gender')
    end
    config.gender = gender;
    clear tmpSpl
    
    if ~exist([currDirSave savFolder],'dir')
        mkdir([currDirSave savFolder]);
    end
    
    dataList = dir(currDirData);
    for i=1:length(dataList)
        cnt = cnt+1;
        waitbar(cnt/NumFiles, h);
        
        currPath = dataList(i).name;
        [~,filename,ext] = fileparts(currPath);
        
        if strcmp(ext, '.wav')
            
            if strcmp(filename(1:3), 'tmp')
                delete([currDirSave currPath]);
                continue;
            end
            
            fprintf('extracting features from %s ...\n',currPath);
            
            sndFilePath = [currDirData currPath];
            outFilePath = [currDirSave savFolder filename '.hdf5'];
            
            extract_features( sndFilePath, outFilePath, config, dirPraat);
            add_deltas( outFilePath, DELTA_WINDOW );

            
        elseif strcmp(ext, '.hdf5')
            if CLEANUP_HDF5
                delete([currDirSave currPath]);
            end
        end
    end
end
delete(h)




