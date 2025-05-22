% get_spikeinfo_bkg_1fov_automatic3_3_RawTrace_3.m
% Batch processing script for spike detection across multiple FOV directories.
% Modifications: Optimized to integrate spike denoising and RawTrace extraction for both spiking and non-spiking neurons.

clear all;
close all;

% ---------------------------
% 1. Add Subfunctions to Path
% ---------------------------
addpath('F:\voltage\20241224  long term voltage imaging\P3-25mW-output\P3-25mWmm2-1\subfunctions\');

% ---------------------------
% 2. Define Main Input Directory
% ---------------------------
mainDir = 'F:\voltage\20241224  long term voltage imaging\P3-50mW-output\test\';

% ---------------------------
% 3. Identify FOV Directories
% ---------------------------
fovPattern = 'P3-50mWmm2-1-FOV*';
fovDirs = dir(fullfile(mainDir, fovPattern));
fovDirs = fovDirs([fovDirs.isdir]);
fovDirs = fovDirs(~ismember({fovDirs.name}, {'.', '..'}));

if isempty(fovDirs)
    error('No FOV directories found with pattern "%s".', fovPattern);
end

% ---------------------------
% 4. Load Shared ROI Data
% ---------------------------
sharedROIFilename = 'ROI_P3-50mWmm2-1.mat';
roiMatFile = fullfile(mainDir, sharedROIFilename);

if ~exist(roiMatFile, 'file')
    error('Shared ROI file "%s" not found in main directory.', sharedROIFilename);
end

try
    tmpROI = load(roiMatFile, 'ROIs');
    ROIs = tmpROI.ROIs;
    clear tmpROI;
catch ME
    error('Failed to load ROIs from "%s": %s', roiMatFile, ME.message);
end

% ---------------------------
% 5. Define Processing Parameters
% ---------------------------
Param.SampleRate = 996;                % Image frame rate (Hz)
Param.SpikeTemplateLength = 3;         % Time window for template generation (time points)
Param.SpikeTemplateN = 1;              % Number of peaks used for template generation
Param.CutOffFreq = 1;                   % High-pass filter cutoff frequency (Hz)
Param.SpikePolarity = 1;                % -1 for negative spikes
Param.MinSpikeTemplateN = [1 1 1 1];    % Minimum number of spikes to calculate template
Param.SNRList = [4 6 9 12];               % Signal-to-noise ratio list
Param.HeadTailSize = 200;                % Head-tail size parameter
Param.CellEnvSize = 15;                  % Size extending from soma to estimate background

% ---------------------------
% 6. Initialize Batch Results Structure
% ---------------------------
BatchResults = struct();

% ---------------------------
% 7. Iterate Through Each FOV Directory
% ---------------------------
for fovIdx = 1:length(fovDirs)
    fovName = fovDirs(fovIdx).name;
    fprintf('Processing %s (%d of %d)...\n', fovName, fovIdx, length(fovDirs));
    
    % Define the path for the current FOV
    fovPath = fullfile(mainDir, fovName);
    
    % Construct FileNamePrefix based on FOV name with suffix '_10_29880'
    FileNamePrefix = [fovName ''];
    
    % Define paths for required files
    tiffFile = fullfile(fovPath, [FileNamePrefix '_51_29801.tif']);
    correctedMatFile = fullfile(fovPath, [FileNamePrefix '_51_29801_corrected.mat']);
    pcaMatFile = fullfile(fovPath, [FileNamePrefix '_51_29801_PCA.mat']);
    
    % Check for the existence of required files
    requiredFiles = {tiffFile, correctedMatFile, pcaMatFile};
    missingFiles = requiredFiles(~cellfun(@(x) exist(x, 'file'), requiredFiles));
    if ~isempty(missingFiles)
        warning('Skipping %s due to missing files:', fovName);
        for mf = 1:length(missingFiles)
            warning(' - %s', missingFiles{mf});
        end
        continue; % Skip to the next FOV
    end
    
    % ---------------------------
    % 8. Load PCA Noise Data
    % ---------------------------
    try
        tmpPCA = load(pcaMatFile, 'NoisePCA');
        NoisePCA = tmpPCA.NoisePCA;
        clear tmpPCA;
    catch ME
        warning('Failed to load NoisePCA from %s: %s', pcaMatFile, ME.message);
        continue;
    end
    
    % ---------------------------
    % 9. Load, Correct, and Create Preview Image
    % ---------------------------
    try
        load(correctedMatFile, 'ImStackCrt');
        ImStack = noise_pca_crt(ImStackCrt, NoisePCA);  % Ensure this function is defined properly
        CrtImg = ones(size(ImStackCrt)) .* mean(ImStackCrt, 3) + double(ImStack);
        
        % Create Preview Image
        numFrames = min(1000, size(ImStack, 3));
        ImPreview = std(single(ImStack(:,:,1:numFrames)), 0, 3);
        ImPreview = ImPreview / max(ImPreview(:));
        ImPreview = repmat(ImPreview, 1, 1, 3);
        
        clear ImStackCrt CrtImg;
    catch ME
        warning('Failed to process image stack or create preview for %s: %s', fovName, ME.message);
        continue;
    end
    
    % ---------------------------
    % 10. Combine All ROIs for Visualization (Optional)
    % ---------------------------
    ROITot = zeros(size(ImPreview,1), size(ImPreview,2));
    for ii = 1:length(ROIs)
        ROITot(ROIs{ii}) = 1;
    end
    
    % ---------------------------
    % 11. Initialize ROIMask and Neuron Structure
    % ---------------------------
    ROIMask = zeros(size(ImPreview,1), size(ImPreview,2));
    Neuron = struct('ROI', {}, 'SpikeInfo', {}, 'RawTrace', {});
    
    % ---------------------------
    % 12. Iterate Through Each ROI
    % ---------------------------
    for ii = 1:length(ROIs)
        Neuron(ii).ROI = ROIs{ii};
        Neuron(ii).SpikeInfo = [];
        Neuron(ii).RawTrace = [];
        if ~isempty(Neuron(ii).ROI)
            fprintf('  Processing ROI #%d/%d in %s...\n', ii, length(ROIs), fovName);
            
            % Reset ROIMask and set current ROI
            ROIMask(:) = 0;
            ROIMask(Neuron(ii).ROI) = 1;
            
            % Extract ROI Data
            try
                [DataROI, SomaIdx, BkgIdx] = extract_roi(Param, ImStack, ROIMask, ROITot);
            catch ME
                warning('    Failed to extract ROI #%d in %s: %s', ii, fovName, ME.message);
                continue;
            end
            
            % Spike Extraction
            try
                SpikeInfo = spike_extract(Param, DataROI, SomaIdx, BkgIdx);
                Neuron(ii).SpikeInfo = SpikeInfo;
                if isempty(SpikeInfo) || isempty(SpikeInfo.SpikeIdx)
                    Neuron(ii).RawTrace = SpikeInfo.RawTrace; % Save RawTrace for non-spiking neurons
                end
            catch ME
                warning('    Failed to extract spikes for ROI #%d in %s: %s', ii, fovName, ME.message);
                continue;
            end
        end
    end
    
    % ---------------------------
    % 13. Count Spike Neurons
    % ---------------------------
    NumberSpikeNeuron = count_spike_neuron(Neuron);
    fprintf('  Number of neurons with detected spikes in %s: %d\n', fovName, NumberSpikeNeuron);
    
    % ---------------------------
    % 14. Save Spike Information
    % ---------------------------
    try
        save(fullfile(fovPath, [FileNamePrefix '_spikeinfo.mat']), 'Neuron', 'Param');
    catch ME
        warning('  Failed to save spike info for %s: %s', fovName, ME.message);
    end
    
    % ---------------------------
    % 15. Store Batch Results
    % ---------------------------
    BatchResults(fovIdx).FOV = fovName; 
    BatchResults(fovIdx).NumberSpikeNeuron = NumberSpikeNeuron;
    
    % ---------------------------
    % 16. Clear Variables for Current FOV
    % ---------------------------
    clear ROITot ROIMask Neuron;
end

% ---------------------------
% 17. Save Batch Processing Summary
% ---------------------------
try
    save(fullfile(mainDir, 'BatchProcessingSummary.mat'), 'BatchResults');
    fprintf('Batch processing completed. Summary saved to BatchProcessingSummary.mat.\n');
catch ME
    warning('Failed to save batch processing summary: %s', ME.message);
end

% ---------------------------
% 18. Display Final Summary
% ---------------------------
for fovIdx = 1:length(BatchResults)
    fprintf('FOV: %s - Spike Neurons: %d\n', BatchResults(fovIdx).FOV, BatchResults(fovIdx).NumberSpikeNeuron);
end
%% Function Definitions

% ---------------------------
% Function: count_spike_neuron
% ---------------------------
function NumberSpikeNeuron = count_spike_neuron(Neuron)
    Count = 0;
    for i = 1:length(Neuron)
        % 多层安全检查
        if isfield(Neuron(i), 'SpikeInfo') && ...              % 检查是否存在SpikeInfo字段
           isstruct(Neuron(i).SpikeInfo) && ...                % 确保是结构体
           isfield(Neuron(i).SpikeInfo, 'SpikeIdx') && ...    % 检查SpikeIdx字段存在
           ~isempty(Neuron(i).SpikeInfo.SpikeIdx)              % 验证SpikeIdx非空
            Count = Count + 1;
        end
    end
    NumberSpikeNeuron = Count;
end

% ---------------------------
% Function: noise_pca_crt
% ---------------------------
function ImgCrt = noise_pca_crt(ImgRaw, NoisePCA)
    NoiseCrtN = 1; % Number of PCA components to correct
    Trace = NoisePCA.Trace(1:NoiseCrtN, :);
    
    ImgSize = size(ImgRaw);
    ImgRaw = reshape(single(ImgRaw), ImgSize(1)*ImgSize(2), ImgSize(3));
    
    % Use backslash operator for numerical stability
    Coef = (Trace * Trace') \ (Trace * ImgRaw');
    
    ImgCrt = ImgRaw - Coef' * Trace;
    ImgCrt = uint16(ImgCrt - min(ImgCrt(:)));
    ImgCrt = reshape(ImgCrt, ImgSize(1), ImgSize(2), ImgSize(3));
end

% ---------------------------
% Function: display_roi
% ---------------------------
function Img = display_roi(Img, ROIMask)
    ROIMaskRGB = repmat(ROIMask, 1, 1, 3);
    ROIMaskRGB = ROIMaskRGB .* rand(1, 1, 3) / 6; % Random color with reduced intensity
    Img = Img + ROIMaskRGB;
end

% ---------------------------
% Function: extract_roi
% ---------------------------
function [DataROI1, SomaIdx, BkgIdx] = extract_roi(Param, ImStack1, ROIMask, ROITot)
    SpikePolarity = Param.SpikePolarity;
    CellEnvSize = Param.CellEnvSize;
    
    % Determine bounding box of the ROI
    [yMax, xMax] = size(ROIMask);
    [rows, cols] = find(ROIMask);
    xmin = max(1, min(cols) - CellEnvSize);
    xmax = min(xMax, max(cols) + CellEnvSize);
    ymin = max(1, min(rows) - CellEnvSize);
    ymax = min(yMax, max(rows) + CellEnvSize);
    
    % Extract ROI data with surrounding environment
    DataROI1 = double(ImStack1(ymin:ymax, xmin:xmax, :));
    
    % Create Soma and Background Masks
    SomaMask = ROIMask(ymin:ymax, xmin:xmax);
    BkgMask = 1 - ROITot(ymin:ymax, xmin:xmax);
    
    % Spike polarity adjustment
    DataROI1 = DataROI1 * SpikePolarity;
    
    % Find indices
    SomaIdx = find(SomaMask(:));
    BkgIdx = find(BkgMask(:));
    
    % Optional: Display masks for verification
    % figure; subplot(1,2,1); imagesc(SomaMask); axis image; title('Soma Mask');
    % subplot(1,2,2); imagesc(BkgMask); axis image; title('Background Mask');
end

% ---------------------------
% Function: spike_extract
% ---------------------------
function SpikeInfo = spike_extract(Param, DataROI1, SomaIdx, BkgIdx)
    % Create Soma Mask (binary)
    SomaMask = zeros(size(DataROI1,1), size(DataROI1,2));
    SomaMask(SomaIdx) = 1;
    
    % High-Pass Filter the ROI Data
    DataROIFilt = highpass_video_filt(Param, DataROI1);
    
    % Reshape filtered data for trace extraction
    DataROIFilt = reshape(DataROIFilt, [], size(DataROIFilt,3));
    
    % Extract Raw Trace (mean over soma pixels)
    RawTrace = mean(DataROIFilt(SomaIdx, :), 1)';
    
    % Extract Background Trace
    Bkg = mean(DataROIFilt(BkgIdx, :), 1)';
    
    % Subtract Background from Raw Trace
    Trace = RawTrace - Bkg;
    
    % Center the Trace
    Trace1 = Trace - mean(Trace(:));
    
    % Estimate Noise Amplitude
    Mask = (Trace1 < 0);
    NoiseAmp1 = sqrt(sum(Trace1.^2 .* Mask) / sum(Mask));
    
    % Assign Trace for Spike Detection
    Trace = Trace1;
    
    % Spike Detection (Assuming spike_denoise is a custom function)
    SpikeInfo = spike_denoise(Param, Trace);
    
    % Optional: Visualization for Verification
    % Note: In batch processing, it's advisable to comment out or save figures instead of displaying them
    % figure(1000);
    % ax1 = subplot(3,1,1); plot(Trace); title('Trace');
    % if ~isempty(SpikeInfo)
    %     hold on; plot(SpikeInfo.SpikeIdx, SpikeInfo.Trace(SpikeInfo.SpikeIdx), 'go');
    %     plot(SpikeInfo.SpikeTemplateIdx, SpikeInfo.Trace(SpikeInfo.SpikeTemplateIdx), 'ro');
    %     hold off;
    % end
    % subplot(3,1,2); plot(SpikeInfo.RawTrace); title('Raw Trace');
    % subplot(3,1,3); plot(SpikeInfo.FiltTrace); title('Filtered Trace');
    
    % Additional visualization can be added as needed
end

% ---------------------------
% Function: highpass_video_filt
% ---------------------------
function VideoFilt = highpass_video_filt(Param, Video)
    CutOffFreq = Param.CutOffFreq;
    SampleRate = Param.SampleRate;
    
    % Normalize frequency
    NormFreq = CutOffFreq / (SampleRate / 2);
    
    % Design Butterworth high-pass filter
    [bb, aa] = butter(3, NormFreq, 'high');
    
    % Permute Video Dimensions to [Time, Y, X] for filtering
    Video = permute(Video, [3, 1, 2]);
    
    % Apply zero-phase filtering
    VideoFilt = filtfilt(bb, aa, Video);
    
    % Permute back to original dimensions [Y, X, Time]
    VideoFilt = permute(VideoFilt, [2, 3, 1]);
end
