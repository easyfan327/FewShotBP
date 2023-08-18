% FECOLS.PTT_PA                       = 1;
% FECOLS.PTT_MAX_ACC                  = 2;
% FECOLS.PTT_MAX_SLP                  = 3;
% FECOLS.PTT_MAX_DACC                 = 4;
% FECOLS.PTT_SYS_PEAK                 = 5;
% FECOLS.PPG_CYCLE                    = 6;     %systolic peak - systolic peak
% FECOLS.ECG_CYCLE                    = 7;     %RR-interval
% FECOLS.AIF                          = 8;     %augmentation index
% FECOLS.LASIF                        = 9;     %large artery stiffness index
% FECOLS.IPA1                         = 10;    %inflection point area ratio 1
% FECOLS.IPA2                         = 11;    %inflection point area ratio 2
% FECOLS.IPAF3                        = 12;    %inflection point area ratio 3
% FECOLS.IPAF4                        = 13;    %inflection point area ratio 4
% FECOLS.AIM                          = 14;    %augmentation index
% FECOLS.LASIM                        = 15;    %large artery stiffness index
% FECOLS.IPAM3                        = 18;    %inflection point area ratio
% FECOLS.IPAM4                        = 19;    %inflection point area ratio
% 
% FECOLS.SDPTG_DA                     = 20;
% FECOLS.SDPTG_BA                     = 21;
% FECOLS.AGI                          = 22;    %aging index

function data = generate_feature_label(data, extracted_annotations, extracted_features, debug)
global ANCOLS FECOLS
%generate_feature_label generates feature labels
% using only ppg / ecg signals
% segmentation by time ticks
config.feature.window_len = 5;
config.feature.window_overlap = 3;

data.mask.overall = data.mask.infinite | data.mask.flat | data.mask.abp_invalid;
invalid_idx = round(data.mask.overall);

total_sec = floor(double(data.len) / double(data.fs));

invalid_sec = sum(reshape(invalid_idx(1:total_sec * round(data.fs)), data.fs, total_sec), 1);

sec_cnt = 0;
for sec_begin = 1 : config.feature.window_overlap : total_sec - config.feature.window_len
    if(sum(invalid_sec(sec_begin : (sec_begin + config.feature.window_len - 1))) < 64)

        data_start = data.fs * (sec_begin - 1) + 1;
        data_end = data_start + config.feature.window_len * data.fs - 1;
        
        time_flag = extracted_annotations(ANCOLS.ECG_RWAVE, :);
        cycle_ids = [];
        for i = 1:length(time_flag)
            if (time_flag(i) > data_start && time_flag(i) < data_end)
                cycle_ids = [cycle_ids, i];
            end
        end
        
        if isempty(cycle_ids)
            continue;
        else
            sec_cnt = sec_cnt + 1;
        
            data.data_range(:, sec_cnt) = [data_start; data_end];
            data.cwtppg(:, :, sec_cnt) = abs(cwt(data.ppg(data_start: data_end)));
            data.cwtecg(:, :, sec_cnt) = abs(cwt(data.ecg(data_start: data_end)));
            data.handcrafted_features(:, sec_cnt) = mean(extracted_features(:, cycle_ids), 2, 'omitnan');
            data.handcrafted_features_std(:, sec_cnt) = std(extracted_features(:, cycle_ids), 0, 2, 'omitnan');
        end

    else
        continue
    end
end

if debug
   figure;
   plot(data.ppg * 20);
   hold on;
   plot(data.abp);
   hold on;
   plot(data.sbp);
   hold on;
   plot(data.dbp);
   hold on;
   scatter(find(data.mask.overall), data.abp(find(data.mask.overall)), 'k.');
%    hold on;
%    for i = 1:size(data.target, 2)
%        seg_start = data.feature(1, i);
%        seg_end = data.feature(2, i);
%        seg_sbp = data.target(1, i);
%        seg_dbp = data.target(2, i);
%        x = seg_start;
%        y = seg_dbp;
%        w = seg_end - seg_start;
%        h = seg_sbp - seg_dbp;
%        rectangle('Position', [x, y, w, h], 'FaceColor', [0, 0, 0, 0], 'EdgeColor', [0, 1, 0, 1]);
%    end
   close;
end
end