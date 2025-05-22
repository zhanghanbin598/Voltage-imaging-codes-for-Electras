%% Final Stable Version Group SNR Calculation Script
clc; clear;

% 1. Parameter settings
mainDir = 'F:\voltage\20241224  long term voltage imaging\P3-50mW-output\P3-50mWmm2-1';
[~, mainFolderName] = fileparts(mainDir);
outputExcel = fullfile(mainDir, [mainFolderName '_Final_SNR.xlsx']);

% 2. Get subfolders
subFolders = dir(mainDir);
subFolders = subFolders([subFolders.isdir] & ~ismember({subFolders.name}, {'.', '..'}));

% 3. Initialize result storage
maxGroups = 8;  % 3 neurons per group
results = cell(0, maxGroups*3);

% 4. Generate column headers
columnNames = cell(1, maxGroups*3);
for g = 1:maxGroups
    columnNames{(g-1)*3 + 1} = ['NeuronID' num2str(g)];
    columnNames{(g-1)*3 + 2} = ['FileID' num2str(g)];
    columnNames{(g-1)*3 + 3} = ['SNR' num2str(g)];
end

% 5. Process subfolders
for f = 1:length(subFolders)
    currentSub = subFolders(f).name;
    spikeFile = fullfile(mainDir, currentSub, [currentSub '_spikeinfo.mat']);
    
    % Handle missing files
    if ~exist(spikeFile, 'file')
        currentRow = cell(1, maxGroups*3);
        currentRow(1:3:end) = arrayfun(@(x)sprintf('Neuron%d',x), 1:maxGroups, 'UniformOutput', false);
        currentRow(2:3:end) = repmat({[mainFolderName '-' currentSub]}, 1, maxGroups);
        results(end+1, :) = currentRow;
        continue;
    end
    
    % 6. Load data
    load(spikeFile, 'Neuron');
    fileID = [mainFolderName '-' currentSub];
    
    % 7. Initialize row
    currentRow = cell(1, maxGroups*3);
    neuronPositions = 1:3:(maxGroups*3);
    
    % 8. Process neurons
    for n = 1:min(length(Neuron), maxGroups)
        pos = neuronPositions(n);
        
        currentRow{pos} = sprintf('Neuron%d', n);
        currentRow{pos+1} = fileID;
        
        try
            [SNR, isValid] = calculateSNR(Neuron(n));
            currentRow{pos+2} = ifelse(isValid, SNR, NaN);
        catch
            currentRow{pos+2} = NaN;
        end
    end
    
    % Fill empty slots
    for n = (length(Neuron)+1):maxGroups
        pos = neuronPositions(n);
        currentRow{pos} = sprintf('Neuron%d', n);
        currentRow{pos+1} = fileID;
        currentRow{pos+2} = NaN;
    end
    
    results(end+1, :) = currentRow;
end

% 9. Handle NaN values
for i = 1:numel(results)
    if isa(results{i}, 'double') && isscalar(results{i}) && isnan(results{i})
        results{i} = [];
    end
end

% 10. Save results
if ~isempty(results)
    T = cell2table(results, 'VariableNames', columnNames);
    writetable(T, outputExcel, 'WriteMode', 'overwrite');
    fprintf('Results saved: %s\n', outputExcel);
else
    fprintf('No valid data found\n');
end

disp('Processing completed!');

%% SNR Calculation Function
function [SNR, isValid] = calculateSNR(Neuron)
    SNR = NaN;
    isValid = false;
    
    try
        if ~isstruct(Neuron) || ~isfield(Neuron, 'SpikeInfo')
            return;
        end
        
        spikeInfo = Neuron.SpikeInfo;
        if ~isstruct(spikeInfo)
            return;
        end
        
        requiredFields = {'FiltTrace', 'SpikeIdx'};
        if ~all(cellfun(@(f) isfield(spikeInfo, f), requiredFields))
            return;
        end
        
        FiltTrace = double(spikeInfo.FiltTrace(:));
        validSpikes = spikeInfo.SpikeIdx(isfinite(spikeInfo.SpikeIdx));
        validSpikes = validSpikes(validSpikes > 0 & validSpikes <= length(FiltTrace));
        
        if isempty(FiltTrace) || isempty(validSpikes)
            return;
        end
        
        % Noise calculation
        noiseMask = true(size(FiltTrace));
        for s = validSpikes'
            winStart = max(1, s-20);
            winEnd = min(length(FiltTrace), s+20);
            noiseMask(winStart:winEnd) = false;
        end
        
        if sum(noiseMask) < 10
            return;
        end
        noiseStd = std(FiltTrace(noiseMask));
        
        % Amplitude calculation
        amplitudes = arrayfun(@(s) abs(FiltTrace(s) - mean(FiltTrace(max(1,s-20):max(1,s-1)))), validSpikes);
        avgAmp = mean(amplitudes);
        
        if noiseStd > 0 && ~isnan(avgAmp)
            SNR = avgAmp / noiseStd;
            isValid = true;
        end
        
    catch
        isValid = false;
    end
end

%% Helper
function out = ifelse(condition, trueVal, falseVal)
    if condition
        out = trueVal;
    else
        out = falseVal;
    end
end
