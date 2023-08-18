clear variables;
close all;

global ANCOLS FECOLS CYCLE_MAX_N ANNO_N FENO_N PAT_MIN FS

addpath("utils\");
addpath("app\");
set_globals();

VISUAL_INSPECT = false

% split data.mat into cases
% dataAll = load("/media/feiyi/ssd/datasets/ucibp/data.mat");
% dataSize = size(dataAll.data);
% outputBaseDirName = '/media/feiyi/ssd/datasets/ucibp/cases/'
% 
% if ~exist(outputBaseDirName, 'dir')
%     mkdir(outputBaseDirName);
% end
% 
% for caseId = 1:dataSize(2)
%     data = dataAll.data{caseId};
%     save(sprintf("%s/case-%d", outputBaseDirName, caseId), 'data', 'caseId', '-v7');
%     fprintf("%d\n", caseId);
% end

% clean
inputBaseDirName = "./cases/*.mat";
outputBaseDirName = "./cases_cleaned/";
if ~exist(outputBaseDirName, 'dir')
    mkdir(outputBaseDirName);
end

file_list = dir(inputBaseDirName);
case_N = size(file_list)

for caseId = 1:case_N(1)

    %% Load the file
    case_file_path = strcat(file_list(caseId).folder, '/', file_list(caseId).name)
    file = load(case_file_path);
    data = preprocess(file.data);
    
    %% fillmissing
    [data.mask.ppg_infinite, data.ppg] = check_infinite(data.ppg, VISUAL_INSPECT);
    [data.mask.abp_infinite, data.abp] = check_infinite(data.abp, VISUAL_INSPECT);
    [data.mask.ecg_infinite, data.ecg] = check_infinite(data.ecg, VISUAL_INSPECT);
    
    data.mask.infinite = data.mask.abp_infinite | data.mask.ppg_infinite | data.mask.ecg_infinite;
    
    infinite_ratio = sum(data.mask.infinite) / double(data.len);

    if (infinite_ratio > 0.05)
        fprintf("case %d, infinite ratio = %f\n", caseId, infinite_ratio);
        continue;
    end
  
    %% check flat lines
    [data.mask.abp_flat, data.d_abp] = check_flat(data.abp, VISUAL_INSPECT);
    [data.mask.ppg_flat, data.d_ppg] = check_flat(data.ppg, VISUAL_INSPECT);
    [data.mask.ecg_flat, data.d_ecg] = check_flat(data.ecg, VISUAL_INSPECT);
    
    data.mask.flat = data.mask.ppg_flat | data.mask.abp_flat | data.mask.ecg_flat;
%    data.mask.flat = data.mask.ppg_flat | data.mask.abp_flat;
    flat_ratio = sum(data.mask.flat) / double(data.len);
    if (flat_ratio > 0.2)
        fprintf("case %d, flat ratio = %f\n", caseId, flat_ratio);
        continue;
    end
    
%     %% filter the signal
%     [b, a] = butter(4, [0.5, 8] / (double(data.fs) / 2));    % butterworth filter
%     data.ppg = filtfilt(b, a, double(data.ppg));             % zero phase filter -> eliminates the phase shift that occurs when filtering
% 

    % Filter ABP using Hampel filter (median, 6 neighbour, 3x standard deviation)
    data.abp = hampel(data.abp, 100, 5);
    [data.mask.abp_invalid, data.sbp, data.dbp] = process_abp(data.abp, VISUAL_INSPECT);
    
    [an, fe, data.ecg, data.ecg_square, data.ppg, vpg, apg] = annotate(data.ecg, data.ppg, data.abp, FS);

    extracted_annotations = an;
    extracted_features = fe;

    data = generate_feature_label(data, extracted_annotations, extracted_features, VISUAL_INSPECT);
    
    % Save results
    if isfield(data, 'data_range')
    
        data = rmfield(data, {'mask', 'd_abp', 'd_ppg', 'd_abp', 'd_ecg'});
        data.ppg = single(data.ppg);
        data.ecg = single(data.ecg);
        data.abp = single(data.abp);
        data.sbp = single(data.sbp);
        data.dbp = single(data.dbp);
        data.cwtppg = single(data.cwtppg);
        data.cwtecg = single(data.cwtecg);
        data.data_range = uint32(data.data_range);
        data.handcrafted_features = single(data.handcrafted_features);
        data.handcrafted_features_std = single(data.handcrafted_features_std);
    
        outputFileName = sprintf(outputBaseDirName + "%d.mat", caseId);
        parsave(outputFileName, data);
        fprintf("case %d generated\n", caseId);
    else
        fprintf("case %d has no valid feature\n", caseId);
    end
    
    if false
        figure;
        subplot(2, 1, 1);
        hold on;
        plot(data.abp, 'b');
        plot(data.sbp, 'r');
        plot(data.dbp, 'r');
        scatter(find(data.mask.abp_infinite), data.abp(find(data.mask.abp_infinite)), 'm.');
        scatter(find(data.mask.abp_flat), data.abp(find(data.mask.abp_flat)), 'y.');
        scatter(find(data.mask.abp_invalid), data.abp(find(data.mask.abp_invalid)), 'k.');
        % add a multiplier to exaggerate the shape
        t = data.ppg * 20;
        plot(t, 'b');
        scatter(find(data.mask.ppg_infinite), t(find(data.mask.ppg_infinite)), 'm.');
        scatter(find(data.mask.ppg_flat), t(find(data.mask.ppg_flat)), 'y.');
        legend('abp', 'sbp', 'dbp', 'abp infinite', 'abp flat', 'abp invalid', 'ppg', 'ppg infinite', 'ppg flat');
%         for i = 1:size(data.target, 2)
%             seg_start = data.feature(1, i);
%             seg_end = data.feature(2, i);
%             seg_sbp = data.target(1, i);
%             seg_dbp = data.target(2, i);
%             x = seg_start;
%             y = seg_dbp;
%             w = seg_end - seg_start;
%             h = seg_sbp - seg_dbp;
%             rectangle('Position', [x, y, w, h], 'FaceColor', [0, 0, 0, 0], 'EdgeColor', [0, 1, 0, 1]);
%         end
        subplot(2, 1, 2);
        hold on;
        plot(data.ecg, 'b');
        scatter(find(data.mask.ecg_infinite), data.ecg(find(data.mask.ecg_infinite)), 'm.');
        scatter(find(data.mask.ecg_flat), data.ecg(find(data.mask.ecg_flat)), 'y.');
        title(sprintf("Case %d", caseId));
        close;
   end
end

function parsave(fname, data)
save(fname, 'data');
end

function [data] = preprocess(input)
data.ppg = input(1,:);
data.abp = input(2,:);
data.ecg = input(3,:);
data.size = size(input);
data.len = data.size(2);
data.fs = 125;
end

function [mask, d] = check_flat(s, debug)
if iscolumn(s)
    s = transpose(s);
end

d = [0, diff(s)];
mask = abs(d) < 1e-6;
for i = 2:length(mask) - 1
    if ~mask(i - 1) && mask(i) && ~mask(i + 1)
        mask(i) = false;
    end
end

if debug
    figure;
    plot(s);
    hold on;
%     plot(d);
%     hold on;
    scatter(find(mask), s(mask), 'r*');
    close;
end
end

function [mask, s_out] = check_infinite(s_in, debug)
mask = ~isfinite(s_in);
s_in(mask) = nan;

[s_out, mask] = fillmissing(s_in, 'linear');
if debug
    figure;
    plot(s_out, 'b');
    hold on;
    scatter(find(mask), s_out(mask), 'r*');
    close;
end
end






