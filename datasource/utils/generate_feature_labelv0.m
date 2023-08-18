function data = generate_feature_label(data, debug)
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
        sec_cnt = sec_cnt + 1;
        
        data_start = data.fs * (sec_begin - 1) + 1;
        data_end = data_start + config.feature.window_len * data.fs - 1;
        
        data.feature(:, sec_cnt) = [data_start; data_end];
        data.cwtppg(:, :, sec_cnt) = abs(cwt(data.ppg(data_start: data_end)));
        data.cwtecg(:, :, sec_cnt) = abs(cwt(data.ecg(data_start: data_end)));
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