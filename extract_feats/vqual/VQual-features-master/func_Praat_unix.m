function [isVoiced, F0, F1, F2, F3, B1, B2, B3, err] = func_Praat_unix(wavfile, windowlength, frameshift, datalen, params, praatPath)                               
%
% Author: Yen-Liang Shue, Speech Processing and Auditory Perception Laboratory, UCLA
% Modified by Kristine Yu 2010-10-16
% Modified by Soo Jin Park 2017-01-23
%
% Copyright UCLA SPAPL 2010

% settings 
% iwantfilecleanup = 1;  %delete files when done

% PraatPath = 'C:\Suzinia\Dropbox\04PhD\Researches\2016_iVector\Toolkits';

minF0           = params.F0.F0min;
maxF0           = params.F0.F0max;
silthres        = params.F0.SilenceThreshold;
voicethres      = params.F0.VoicedThreshold ;
octavecost      = params.F0.OctaveCost;
octavejumpcost  = params.F0.OctaveJumpCost;
voiunvoicost    = params.F0.VoicedUnvoicedCost;
killoctavejumps = params.F0.KillOctaveJumps;
smooth          = params.F0.Smooth;
smoothbw        = params.F0.SmoothingBandwidth;
interpolate     = params.F0.Interpolate;
method          = params.F0.Method; % set cross-correlation as default for Praat f0 estimation
%frameprecision  = params.F0.FramePrecision;
maxFormant    = params.Formants.MaxFormantFreq;
num_formants  = params.Formants.NumFormants;




% if ispc      
%     cmd_v = sprintf(['"%sPraat_windows.exe" --run %spraatF0.praat "%s" %.3f ' ...
%         '%.3f %.3f %.3f %.3f %.3f %.3f %.3f %d %d %.3f %d "%s"'],...
%         praatPath, praatPath, wavfile, frameshift, minF0, maxF0, ...
%         silthres, voicethres, octavecost, octavejumpcost, voiunvoicost, ...
%         killoctavejumps, smooth, smoothbw, interpolate, method);
%     
%     % set silence threshold and voiced treshold zero to calculate pitch
%     cmd_f0 = sprintf(['"%sPraat_windows.exe" --run %spraatF0.praat "%s" %.3f ' ...
%         '%.3f %.3f %.3f %.3f %.3f %.3f %.3f %d %d %.3f %d "%s"'],...
%         praatPath, praatPath, wavfile, frameshift, minF0, maxF0, ...
%         0, 0, octavecost, octavejumpcost, voiunvoicost, ...
%         killoctavejumps, smooth, smoothbw, interpolate, method);
%     
% 
%     cmd_fmt = sprintf('"%sPraat_windows.exe" --run %spraatformants.praat "%s" %.3f %.3f %.1f %d', ...
%         praatPath, praatPath, wavfile, frameshift, windowlength, num_formants, maxFormant);
%     
% elseif isunix

    
    cmd_v = sprintf(['%sPraat %spraatF0.praat %s %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %d %d %.3f %d %s'], ...
        praatPath, praatPath,wavfile, frameshift, minF0, maxF0, silthres, voicethres, octavecost, octavejumpcost, voiunvoicost, killoctavejumps, smooth, smoothbw, interpolate, method);
    
    
    cmd_f0 = sprintf(['%sPraat %spraatF0.praat %s %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %d %d %.3f %d %s'], ...
        praatPath, praatPath, wavfile, frameshift, minF0, maxF0, 0, 0, octavecost, octavejumpcost, voiunvoicost, killoctavejumps, smooth, smoothbw, interpolate, method);
    
    cmd_fmt = sprintf(['%sPraat %spraatformants.praat ' ...
        '%s %.3f %.3f %.1f %d'], praatPath, praatPath, wavfile, frameshift, windowlength, num_formants, maxFormant);
    

% else % otherwise  
%     error('unsupported OS')
% end

fileName_F0  = [wavfile '.praat' method];
fileName_Formant = [wavfile '.pfmt'];

% num_formants = ceil(num_formants);

num_outputFmts = ceil(num_formants);


%% Initialize
% NOTE: Initialize the value before reading the features
% to prevent returning without assigning values to the output variables
isVoiced = false(datalen,1);
F0  = zeros(datalen, 1) * NaN;

F1 = zeros(datalen, 1) * NaN;  B1 = zeros(datalen, 1) * NaN;
F2 = zeros(datalen, 1) * NaN;  B2 = zeros(datalen, 1) * NaN;
F3 = zeros(datalen, 1) * NaN;  B3 = zeros(datalen, 1) * NaN;


% if ispc
%     % run Praat
%     err1 = system(cmd_v);
%     
%     % Read from file
%     fid = fopen(fileName_F0, 'rt');
%     C_v = textscan(fid, '%f %f', 'delimiter', '\n', 'TreatAsEmpty', '--undefined--');
%     fclose(fid);
%     delete(fileName_F0);
% 
%     % run
%     err2 = system(cmd_f0);
%     
%     % Read from file
%     fid = fopen(fileName_F0, 'rt');
%     C = textscan(fid, '%f %f', 'delimiter', '\n', 'TreatAsEmpty', '--undefined--');
%     fclose(fid);
%     
%     % run
%     err3 = system(cmd_fmt);
%     
%     % Read from file
%     fid = fopen(fileName_Formant, 'rt');
%     
%     %read and discard the header
%     textscan(fid, '%s', 1, 'delimiter', '\n');
%     
%     % read the rest
%     str = ['%f %f ' repmat('%f ', [1, 2*num_outputFmts])];
%     C_fmt = textscan(fid, str, 'delimiter', '\n', 'TreatAsEmpty', '--undefined--');
%     fclose(fid);
% else
%     
    % VAD
    err1 = unix(cmd_v);
    fid = fopen(fileName_F0, 'rt');
    C_v = textscan(fid, '%f %f', 'delimiter', '\n', 'TreatAsEmpty', '--undefined--');
    fclose(fid);
    delete(fileName_F0);
    
    if isempty(C_v{1})
        err = false;
        return;
    end
    
    % F0
    err2 = unix(cmd_f0);
    fid = fopen(fileName_F0, 'rt');
    C = textscan(fid, '%f %f', 'delimiter', '\n', 'TreatAsEmpty', '--undefined--');
    fclose(fid);
    delete(fileName_F0);
    
    % Formants
    err3 = unix(cmd_fmt);
    fid = fopen(fileName_Formant, 'rt');
    textscan(fid, '%s', 1, 'delimiter', '\n'); %read and discard the header
    str = ['%f %f ' repmat('%f ', [1, 2*num_outputFmts])]; % read the rest
    C_fmt = textscan(fid, str, 'delimiter', '\n', 'TreatAsEmpty', '--undefined--');
    fclose(fid);
    delete(fileName_Formant);
% end



% if exist(fileName_F0, 'file')
%     delete(fileName_F0);
% end
% if exist(fileName_Formant, 'file')
%     delete(fileName_Formant);
% end


err=err1 || err2 || err3;
if err % oops, error, exit now

    F0 = NaN;
    
    F1 = NaN; F2 = NaN; F3 = NaN; %F4 = NaN; F5 = NaN; F6 = NaN; F7 = NaN;
    B1 = NaN; B2 = NaN; B3 = NaN; %B4 = NaN; B5 = NaN; B6 = NaN; B7 = NaN;
    
    return;
end






%%
timePrecision = params.F0.FramePrecision * frameshift * 1000;


%KY Since Praat outputs no values in silence/if no f0 value returned, must pad with leading
% undefined values if needed, so we set start time to 0, rather than t(1)

% start = 0;
% finish = t(end);
increment = frameshift * 1000;


t  = round(C_v{1} * 1000);  % time locations from praat pitch track
for k=0:increment:t(end)
    [~, inx] = min(abs(t - k)); % try to find the closest value
    if abs(t(inx) - k) > timePrecision  % no valid value found
        continue;
    end
    
    n = round(k / (frameshift * 1000)) + 1; % KY I added a one since
    % Matlab indexing starts at 1
    % not 0
    
    try
        isVoiced(n)= true;
    catch
        if n < 1 || n > datalen
            continue;
        else
            warning('Prrat:error in assigning value');
        end
    end
    
end


t  = round(C{1} * 1000);  % time locations from praat pitch track
for k=0:increment:t(end)
    [~, inx] = min(abs(t - k)); % try to find the closest value
    if abs(t(inx) - k) > timePrecision  % no valid value found
        continue;
    end
    
    n = round(k / (frameshift * 1000)) + 1; % KY I added a one since
                                            % Matlab indexing starts at 1
                                            % not 0

    if n < 1 || n > datalen
        continue;
    end

    try
        F0(n) = C{2}(inx);
    catch

            warning('Prrat:error in assigning value');

    end
end


t  = round(C_fmt{1} * 1000);  % time locations from praat pitch track
for k=0:increment:t(end)
    [~, inx] = min(abs(t - k)); % try to find the closest value
    if abs(t(inx) - k) > timePrecision  % no valid value found
        continue;
    end
    
    n = round(k / (frameshift * 1000)) + 1; % KY I added a one since
                                            % Matlab indexing starts at 1
                                            % not 0
    if n < 1 || n > datalen
        continue;
    end
      
    try
        F1(n) = C_fmt{3}(inx);
        B1(n) = C_fmt{4}(inx);
        F2(n) = C_fmt{5}(inx);
        B2(n) = C_fmt{6}(inx);
        F3(n) = C_fmt{7}(inx);
        B3(n) = C_fmt{8}(inx);
    catch
        warning('Prrat:error in assigning value');
    end
end
% isVoiced = ~isnan(vF0);

end
