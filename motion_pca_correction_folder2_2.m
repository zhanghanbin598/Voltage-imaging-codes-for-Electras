% process_images.m
% This script batch processes all TIFF files within each subfolder of the specified
% input directory. For each TIFF file, it performs motion estimation, motion correction,
% denoising, and Principal Component Analysis (PCA). The results are saved as .mat files
% within the respective subfolders.

clear all;          % Clear all variables from the workspace
close all;          % Close all open figure windows
addpath("F:\voltage\20241224  long term voltage imaging\P3-25mW-output\P3-25mWmm2-1\subfunctions/");  % Add subfunctions to MATLAB path

%% *** 1. Define Input Directory ***
% Update this path to the root input directory containing all subfolders
InputDir = 'F:\voltage\5mWmm2\20240803 intermitent imaging\20250223-output\';

% Check if the input directory exists
if ~isfolder(InputDir)
    error('Input directory "%s" does not exist.', InputDir);
end

%% *** 2. Define PCA Block Size ***
% Define block size for PCA (ensure BNxy divides the image dimensions)
BNxy = 8;

%% *** 3. Detect All Subfolders ***
% Get a list of all subfolders in the input directory
subfolders = dir(InputDir);
subfolders = subfolders([subfolders.isdir]); % Keep only directories
subfolders = subfolders(~ismember({subfolders.name}, {'.', '..'})); % Exclude '.' and '..'

% Check if there are any subfolders
if isempty(subfolders)
    error('No subfolders found in the input directory "%s".', InputDir);
end

fprintf('Found %d subfolder(s) in the input directory.\n\n', length(subfolders));

%% *** 4. Loop Through Each Subfolder and Process TIFF Files ***
for i = 1:length(subfolders)
    currentSubfolderName = subfolders(i).name;
    currentSubfolderPath = fullfile(InputDir, currentSubfolderName);
    
    fprintf('Processing Subfolder %d of %d: %s\n', i, length(subfolders), currentSubfolderName);
    
    % *** a. Detect TIFF Files in the Current Subfolder ***
    tiffFiles = dir(fullfile(currentSubfolderPath, '*.tif'));
    numTiffFiles = length(tiffFiles);
    
    if numTiffFiles == 0
        fprintf('  No .tif files found in %s. Skipping this subfolder.\n\n', currentSubfolderName);
        continue; % Skip to the next subfolder
    end
    
    % *** b. Process Each TIFF File in the Current Subfolder ***
    for j = 1:numTiffFiles
        tiffFileName = tiffFiles(j).name;
        tiffFilePath = fullfile(currentSubfolderPath, tiffFileName);
        
        fprintf('  Processing TIFF File %d of %d: %s\n', j, numTiffFiles, tiffFileName);
        
        % *** i. Define FileNamePrefix (Base Name without Extension) ***
        [~, FileNamePrefix, ~] = fileparts(tiffFileName);
        
        try
            %% *** ii. Read the Image Stack ***
            ImStack = imstackread(tiffFilePath);
            fprintf('    Image stack read successfully.\n');
            
            %% *** iii. Generate Motion Correction Template and Estimate Shifts ***
            Template = single(mean(double(ImStack), 3)); % Ensure Template is single
            
            % Define Region of Interest (ROI) [y0, x0, y_width, x_width]
            ROIPos = [1, 1, size(ImStack, 1), size(ImStack, 2)]; % Entire image
            
            % Estimate motion shifts using the motion_est function
            Shift = motion_est(Template, ROIPos, ImStack);
            fprintf('    Motion estimation completed.\n');
            
            % Save Motion Estimation Data
            motionEstFileName = [FileNamePrefix, '_motion_estimation.mat'];
            motionEstFilePath = fullfile(currentSubfolderPath, motionEstFileName);
            save(motionEstFilePath, 'Template', 'ROIPos', 'Shift', '-v7'); % Specify .mat version
            fprintf('    Saved %s_motion_estimation.mat\n', FileNamePrefix);
            
            %% *** iv. Apply Motion Correction and Denoise Image Stack ***
            ImStackCrt = motion_crt(ImStack, Shift);
            fprintf('    Motion correction applied.\n');
            
            % Combine consecutive frames to enhance signal (optional)
            ImStackCrt = ImStackCrt(:, :, 2:end) + ImStackCrt(:, :, 1:end-1);
            fprintf('    Consecutive frames combined.\n');
            
            % Save Corrected Image Stack
            correctedFileName = [FileNamePrefix, '_corrected.mat'];
            correctedFilePath = fullfile(currentSubfolderPath, correctedFileName);
            save(correctedFilePath, 'ImStackCrt', '-v7.3'); % '-v7.3' for large files
            fprintf('    Saved %s_corrected.mat\n', FileNamePrefix);
            
            %% *** v. Perform PCA on the Corrected Image Stack ***
            ImPCA = [];
            
            % Crop the image stack to be divisible by BNxy
            Tmp = ImStackCrt(1:floor(size(ImStackCrt,1)/BNxy)*BNxy, ...
                            1:floor(size(ImStackCrt,2)/BNxy)*BNxy, :);
            
            % Reshape the image stack for block-wise averaging
            Tmp = reshape(Tmp, BNxy, size(Tmp,1)/BNxy, BNxy, size(Tmp,2)/BNxy, []);
            ImPCA(:,:,:) = squeeze(mean(mean(single(Tmp), 1), 3));
            
            % Extract noise PCA components
            NoisePCA = noise_pca_extract(ImPCA);
            fprintf('    PCA extraction completed.\n');
            
            % Save PCA Results
            pcaFileName = [FileNamePrefix, '_PCA.mat'];
            pcaFilePath = fullfile(currentSubfolderPath, pcaFileName);
            save(pcaFilePath, 'NoisePCA', 'BNxy', 'ImPCA', '-v7'); % Specify .mat version
            fprintf('    Saved %s_PCA.mat\n', FileNamePrefix);
            
            fprintf('    Successfully processed %s.\n\n', tiffFileName);
            
        catch ME
            fprintf('    Error processing %s: %s. Skipping this file.\n\n', tiffFileName, ME.message);
            continue; % Skip to the next file
        end
    end
end

fprintf('All subfolders have been processed successfully.\n');

%% *** 5. Function Definitions ***

% Function to extract PCA components from noise
function NoisePCA = noise_pca_extract(Img)
    CmpN = 50; % Number of principal components to extract
    
    % Permute and reshape the image data for PCA
    Img = permute(Img, [1, 2, 4, 3]);
    Img = reshape(Img, size(Img,1), [], size(Img,4));
    ImgSize = size(Img);
    Img = reshape(Img, ImgSize(1)*ImgSize(2), []);
    
    %% Perform Singular Value Decomposition (SVD) for PCA
    tic
    [U, S, V] = svds(Img, CmpN);
    toc
    
    %% Reshape PCA Patterns and Compute Trace
    Patterns = reshape(U, ImgSize(1), ImgSize(2), []);
    Trace = S * V';
    clear S V
    
    %% Optional: Visualize PCA Patterns and Trace
    for ii = 1:size(Patterns,3)
        figure(1);
        subplot(1,2,1);
        imagesc(Patterns(:,:,ii));
        axis image;
        colorbar;
        title(['Pattern ' num2str(ii)]);
        
        subplot(1,2,2);
        plot(Trace(ii,:));
        title(['Trace ' num2str(ii)]);
        
        % Uncomment the next line to pause between plots
        % pause;
    end
    
    % Store PCA results in a structure
    NoisePCA.Patterns = Patterns;
    NoisePCA.Trace = Trace;
end

% Function to estimate motion shifts
function Shift = motion_est(Template, ROIPos, ImStack)
    MaxShift = 4; % Maximum allowed shift in pixels
    
    % Extract ROI from the image stack and template
    ROI = ImStack(ROIPos(1):ROIPos(1)+ROIPos(3)-1, ...
                 ROIPos(2):ROIPos(2)+ROIPos(4)-1, :);
    ROI = double(ROI);
    TemplateROI = Template(ROIPos(1):ROIPos(1)+ROIPos(3)-1, ...
                           ROIPos(2):ROIPos(2)+ROIPos(4)-1);
    
    % Create a mask to ignore borders where shifts exceed MaxShift
    Mask = zeros(size(ROI,1), size(ROI,2));
    Mask(MaxShift+1:end-MaxShift, MaxShift+1:end-MaxShift) = 1;
    
    % Compute complex representation for cross-correlation
    ROI_complex = (ROI - circshift(ROI, [1, 0, 0])) + 1i*(ROI - circshift(ROI, [0, 1, 0]));
    TemplateROI_complex = (TemplateROI - circshift(TemplateROI, [1, 0])) + 1i*(TemplateROI - circshift(TemplateROI, [0, 1]));
    
    % Optional: Visualize Mask and Template ROI
    figure(1); imagesc(Mask); axis image; title('Mask');
    figure(2); imagesc(abs(TemplateROI_complex)); axis image; title('Template ROI Complex Magnitude');
    
    % Apply mask to ROI and Template ROI
    ROIRaw = ROI_complex .* Mask;
    TemplateROI_complex = TemplateROI_complex .* Mask;
    
    % Perform cross-correlation in frequency domain
    XCorr = abs(fftshift(fftshift(ifft2(fft2(TemplateROI_complex) .* conj(fft2(ROIRaw))), 1), 2));
    
    % Compute normalization reference
    Ref = sqrt(abs(fftshift(fftshift(ifft2(fft2(Mask) .* conj(fft2(abs(ROIRaw).^2))), 1), 2)));
    Ref = Ref .* sqrt(abs(fftshift(fftshift(ifft2(fft2(abs(TemplateROI_complex).^2) .* conj(fft2(Mask))), 1), 2)));
    XCorr = XCorr ./ (Ref + eps); % Avoid division by zero
    
    % Define the center of cross-correlation
    Center = floor(size(XCorr)/2) + 1;
    
    % Extract a region around the center based on MaxShift
    XCorrROI = XCorr(Center(1)-MaxShift : Center(1)+MaxShift, ...
                    Center(2)-MaxShift : Center(2)+MaxShift, :);
    
    % Reshape for peak detection
    Tmp = reshape(XCorrROI, [], size(XCorrROI,3));
    [XCorrMax, Idx] = max(Tmp, [], 1);
    [YIdx, XIdx] = ind2sub([size(XCorrROI,1), size(XCorrROI,2)], Idx);
    
    % Fine peak fitting for subpixel accuracy
    PeakFitSize = 2;
    [xx, yy] = meshgrid(-PeakFitSize:PeakFitSize, -PeakFitSize:PeakFitSize);
    YIdxFine = zeros(1, size(XCorr,3));
    XIdxFine = zeros(1, size(XCorr,3));
    
    for ii = 1:size(XCorr,3)
        Tmp_patch = XCorrROI(YIdx(ii)-PeakFitSize : YIdx(ii)+PeakFitSize, ...
                             XIdx(ii)-PeakFitSize : XIdx(ii)+PeakFitSize, ii);
        Tmp_patch = Tmp_patch - min(Tmp_patch(:)); % Normalize
        
        YIdxFine(ii) = sum(sum(yy .* Tmp_patch)) / sum(Tmp_patch(:)) + YIdx(ii);
        XIdxFine(ii) = sum(sum(xx .* Tmp_patch)) / sum(Tmp_patch(:)) + XIdx(ii);
    end
    
    % Compute shifts relative to the center
    YShift = YIdx - MaxShift - 1;
    XShift = XIdx - MaxShift - 1;
    
    YShiftFine = YIdxFine - MaxShift - 1;
    XShiftFine = XIdxFine - MaxShift - 1;
    
    % Optional: Plot shift information
    figure(100); plot([YShift(:), XShift(:), YShiftFine(:), XShiftFine(:)]);
    title('Shift Estimates');
    legend('YShift', 'XShift', 'YShiftFine', 'XShiftFine');
    
    figure(101); plot(XCorrMax);
    title('Cross-Correlation Maximums');
    
    % Store shifts in a structure and cast to single
    Shift.YShift = single(YShiftFine);
    Shift.XShift = single(XShiftFine);
end

% Function to correct motion based on estimated shifts
function ImgCrt = motion_crt(Img, Shift)
    MaxShift = 1; % Maximum shift to apply (can be adjusted based on data)
    
    % Extract fine shifts
    XShiftFine = Shift.XShift;
    YShiftFine = Shift.YShift;
    TN = length(XShiftFine); % Number of frames to correct
    
    % Initialize corrected image stack
    ImgCrt = double(Img(:, :, end-TN+1:end));
    
    % Create coordinate grids
    [x0, y0] = meshgrid(1:size(Img,2), 1:size(Img,1));
    
    % Apply shifts to each frame
    for ii = 1:length(XShiftFine)
        ImgCrt(:,:,ii) = interp2(x0, y0, ImgCrt(:,:,ii), ...
                                 x0 - XShiftFine(ii), y0 - YShiftFine(ii), ...
                                 'linear', 0);
    end
    
    % Apply mask to remove edge artifacts
    Mask = zeros(size(ImgCrt,1), size(ImgCrt,2));
    Mask(MaxShift+1:end-MaxShift, MaxShift+1:end-MaxShift) = 1;
    ImgCrt = uint16(ImgCrt .* Mask);
end
