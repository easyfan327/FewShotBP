function [mask, upper_env, lower_env] = process_abp(s, debug)
config.spectrogram.window = 500;
config.spectrogram.overlap = 250;
config.spectrogram.fftn = 1024;
config.spectrogram.fs = 125;
config.spectrogram.peakfinding_bias = 5;

config.envelope.abp_peak_min_prominence = 5;
config.envelope.zero = 1e-4;
config.envelope.error_tolerance = 3;
config.envelope.fluctuation_tolerance = 3;
config.envelope.fluctuation_eval_length = 20;
%% s to be a column vector
needsTranspose = isrow(s);
if needsTranspose
    s = s(:);
end

%% use spectrogram to estimate the pulse interval timespan in ABP waveform
[spec, freq, time] = spectrogram(s, config.spectrogram.window, config.spectrogram.overlap, config.spectrogram.fftn, config.spectrogram.fs);

[val, idx] = max(abs(spec(config.spectrogram.peakfinding_bias:end, :)), [], 1);
idx = idx + config.spectrogram.peakfinding_bias - 1;

if debug
    image(abs(spec));
    hold on;
    scatter(1:length(time), idx, 'r*');
    close;
end

%% use average cycle length to determine the minimal peak distance
avg_cycle_length = config.spectrogram.fs / mean(freq(idx));

[upper_env, lower_env, up, lp] = evaluate_envelope(s, avg_cycle_length - 10, config.envelope.abp_peak_min_prominence);

%% remove the signal where
% upper envelope is lower than the actual upper bound
% lower envelope is higher than the actual lower bound
% the lower envelope is lower than zero
mask = (upper_env + config.envelope.error_tolerance) < s | (lower_env - config.envelope.error_tolerance) > s | lower_env < 0 | upper_env < 0;
std_upper_env = movstd(upper_env, config.envelope.fluctuation_eval_length);
std_lower_env = movstd(lower_env, config.envelope.fluctuation_eval_length);
mask2 = abs(std_upper_env) > config.envelope.fluctuation_tolerance | abs(std_lower_env) > config.envelope.fluctuation_tolerance;
if debug
    figure;
    plot(s);
    hold on;
    plot(upper_env);
    hold on;
    scatter(up, s(up), 'r.');
    hold on;
    plot(lower_env);
    hold on;
    scatter(lp, s(lp), 'b.');
    hold on;
    plot(std_upper_env);
    hold on;
    plot(std_lower_env);
    hold on;
    scatter(find(mask), s((find(mask))));
    hold on;
    scatter(find(mask2), s((find(mask2))), 'k*');
    close;
end
mask = mask | mask2;
if needsTranspose
    upper_env = transpose(upper_env);
    lower_env = transpose(lower_env);
    mask = transpose(mask);
end
end