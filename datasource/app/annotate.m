%% annotate the critical features/ points in ECG/PPG/ABP signals
function [an, fe, ecg, ecg_square, ppg, vpg, apg] = annotate(ecg, ppg, abp, fs)
global ANCOLS FECOLS CYCLE_MAX_N ANNO_N FENO_N PAT_MIN

% filter the signal
wt = modwt(ecg, 5);
wtrec = zeros(size(wt));
wtrec(3:5,:) = wt(3:5,:);
ecg_filtered = imodwt(wtrec,'sym4');

[b, a] = butter(4, [0.5, 8] / (double(125) / 2));
ppg_filtered = filtfilt(b, a, ppg);

ecg = ecg_filtered;
ppg = ppg_filtered;
ecg_square = abs(ecg_filtered) .* 2;

% initial parameter estimation
[rwave_amp_th, rwave_interval_th, ppg_cycle] = estimate_initial_threshold(ecg_square, ppg, fs, 1024);
ecg_cycle = ppg_cycle;

ppg_cycle = update_ppg_cycle(ppg_cycle, true);
ecg_cycle = update_ecg_cycle(ecg_cycle, true);

% get vpg and apg
vpg = diff([ppg, ppg(end)]);
apg = diff([vpg, vpg(end)]);

an = zeros(ANNO_N, CYCLE_MAX_N);
fe = zeros(FENO_N, CYCLE_MAX_N);
cycle_cnt = 1;
while(1)

    %% find R-wave in ECG
    if cycle_cnt == 1
        %
        %cycle_start_rwave = get_rwave(ecg_square, 2: 2 + ppg_cycle * 3, rwave_amp_th);
        cycle_start_rwave = select_extremum(ecg_square, 2: 2 + ecg_cycle * 2, "max", "max", rwave_amp_th, "larger");
    else
        % find R-wave in estimated range
        % [previous_rwave + rwave_interval_th -> previous_rwave + rwave_interval_th + ppg_cycle]
        previous_rwave = an(ANCOLS.ECG_RWAVE, cycle_cnt - 1);
        range = floor(previous_rwave + rwave_interval_th): floor(previous_rwave + rwave_interval_th + ecg_cycle);
        %cycle_start_rwave = get_rwave(ecg_square, range, rwave_amp_th);
        cycle_start_rwave = select_extremum(ecg_square, range, "max", "max", rwave_amp_th, "larger");


        % in case the R-wave is not found
        % search [previous_rwave + rwave_interval_th -> end]
        if cycle_start_rwave == 0 || isnan(cycle_start_rwave)
            range = floor(previous_rwave + rwave_interval_th): length(ecg_square) - 1;
            %cycle_start_rwave = get_rwave_alt(ecg_square, range, rwave_amp_th);
            cycle_start_rwave = select_extremum(ecg_square, range, "max", "first", rwave_amp_th, "larger");
        end

        an(ANCOLS.DBG_RWAVE_FIND_START, cycle_cnt) = range(1);
        an(ANCOLS.DBG_RWAVE_FIND_END, cycle_cnt) = range(end);
        

        % no R-wave in [previous_rwave + rwave_interval_th -> end]
        if cycle_start_rwave == 0 || isnan(cycle_start_rwave)
            break;
        else
            % update the amplitude threshold for R-wave searching
            rwave_amp_th_new = ecg_square(cycle_start_rwave) / 4;
            rwave_amp_th = rwave_amp_th * 0.8 +  rwave_amp_th_new * 0.2;
        end
    end

    if cycle_start_rwave + rwave_interval_th + ppg_cycle > length(ecg) || cycle_start_rwave + rwave_interval_th > length(ecg)
        break;
    end

    %% save R-wave annotation
    an(ANCOLS.ECG_RWAVE, cycle_cnt) = cycle_start_rwave;

    %% PPG annotations
    if cycle_cnt >= 2
        start = max([cycle_start_rwave + PAT_MIN, ...
            an(ANCOLS.PPG_DIA_IFPM, cycle_cnt - 1), ...
            an(ANCOLS.PPG_DIA_PAM, cycle_cnt - 1), ...
            an(ANCOLS.PPG_DIA_PEAKM, cycle_cnt - 1), ...
            an(ANCOLS.PPG_SYS_PA, cycle_cnt - 1) - an(ANCOLS.ECG_RWAVE, cycle_cnt - 1)], [], 'omitnan');
    else
        start = cycle_start_rwave + PAT_MIN;
    end

    % PPG systolic pulse arrival point
    range = start: cycle_start_rwave + PAT_MIN + ppg_cycle;
    an(ANCOLS.PPG_SYS_PA, cycle_cnt) = select_extremum(ppg, range, "min", "min");
    fe(FECOLS.PTT_PA, cycle_cnt) = (an(ANCOLS.PPG_SYS_PA, cycle_cnt) - an(ANCOLS.ECG_RWAVE, cycle_cnt));

    % PPG systolic pulse peak
    range = an(ANCOLS.PPG_SYS_PA, cycle_cnt): floor(an(ANCOLS.PPG_SYS_PA, cycle_cnt) + 0.5 * ppg_cycle);
    an(ANCOLS.PPG_SYS_PEAK, cycle_cnt) = select_extremum(ppg, range, "max", "max");
    fe(FECOLS.PTT_SYS_PEAK, cycle_cnt) = (an(ANCOLS.PPG_SYS_PEAK, cycle_cnt) - an(ANCOLS.ECG_RWAVE, cycle_cnt));
    %dbg start
    an(ANCOLS.DBG_SYS_PEAK_FIND_START, cycle_cnt) = an(ANCOLS.PPG_SYS_PA, cycle_cnt);
    an(ANCOLS.DBG_SYS_PEAK_FIND_END, cycle_cnt) = floor(an(ANCOLS.PPG_SYS_PA, cycle_cnt) + 0.5 * ppg_cycle);
    %dbg end

    if ~isnan(an(ANCOLS.PPG_SYS_PEAK, cycle_cnt))
        range = an(ANCOLS.PPG_SYS_PA, cycle_cnt):an(ANCOLS.PPG_SYS_PEAK, cycle_cnt);
        an(ANCOLS.PPG_SYS_ASCEND_MAX_SLP, cycle_cnt) = select_extremum(vpg, range, "max", "max");
        fe(FECOLS.PTT_MAX_SLP, cycle_cnt) = (an(ANCOLS.PPG_SYS_ASCEND_MAX_SLP, cycle_cnt) - an(ANCOLS.ECG_RWAVE, cycle_cnt));
        an(ANCOLS.PPG_SYS_ASCEND_MAX_ACC, cycle_cnt) = select_extremum(apg, range, "max", "max");
        fe(FECOLS.PTT_MAX_ACC, cycle_cnt) = (an(ANCOLS.PPG_SYS_ASCEND_MAX_ACC, cycle_cnt) - an(ANCOLS.ECG_RWAVE, cycle_cnt));
        an(ANCOLS.PPG_SYS_ASCEND_MAX_DACC, cycle_cnt) = select_extremum(apg, range, "min", "min");
        fe(FECOLS.PTT_MAX_DACC, cycle_cnt) = (an(ANCOLS.PPG_SYS_ASCEND_MAX_DACC, cycle_cnt) - an(ANCOLS.ECG_RWAVE, cycle_cnt));
        %% estimate cyle end
        range = an(ANCOLS.PPG_SYS_PEAK, cycle_cnt):an(ANCOLS.PPG_SYS_PEAK, cycle_cnt) + ppg_cycle;
        cycle_minima = select_extremum(ppg, range, "min", "min");
        estimated_cycle_end = min( ...
            [floor(an(ANCOLS.PPG_SYS_PEAK, cycle_cnt) + ppg_cycle - (an(ANCOLS.PPG_SYS_PEAK, cycle_cnt) - an(ANCOLS.PPG_SYS_PA, cycle_cnt))), ...
            floor(an(ANCOLS.PPG_SYS_PA, cycle_cnt) + ppg_cycle), ...
            cycle_minima, ...
            length(ecg)-1]);

        %% find SDPTG features
        if ~isnan(an(ANCOLS.PPG_SYS_ASCEND_MAX_DACC, cycle_cnt)) && ~isnan(an(ANCOLS.PPG_SYS_ASCEND_MAX_ACC, cycle_cnt))
            range = an(ANCOLS.PPG_SYS_ASCEND_MAX_DACC, cycle_cnt): estimated_cycle_end;
            an(ANCOLS.SDPTG_C, cycle_cnt) = select_extremum(apg, range, "max", "first");
            range =  an(ANCOLS.SDPTG_C, cycle_cnt): estimated_cycle_end;
            an(ANCOLS.SDPTG_D, cycle_cnt) = select_extremum(apg, range, "min", "first");
            range =  an(ANCOLS.SDPTG_D, cycle_cnt): estimated_cycle_end;
            an(ANCOLS.SDPTG_E, cycle_cnt) = select_extremum(apg, range, "max", "first");

            
            sdptg_a = get_height(apg, an(ANCOLS.SDPTG_A, cycle_cnt));
            sdptg_b = get_height(apg, an(ANCOLS.SDPTG_B, cycle_cnt));
            sdptg_c = get_height(apg, an(ANCOLS.SDPTG_C, cycle_cnt));
            sdptg_d = get_height(apg, an(ANCOLS.SDPTG_D, cycle_cnt));
            sdptg_e = get_height(apg, an(ANCOLS.SDPTG_E, cycle_cnt));

            fe(FECOLS.SDPTG_BA, cycle_cnt) = abs(sdptg_b) / abs(sdptg_a);
            fe(FECOLS.SDPTG_DA, cycle_cnt) = abs(sdptg_d) / abs(sdptg_a);
            fe(FECOLS.AGI, cycle_cnt) = (sdptg_b - sdptg_c - sdptg_d - sdptg_e) / sdptg_a;

        end

        range = an(ANCOLS.PPG_SYS_PEAK, cycle_cnt):estimated_cycle_end;
        an(ANCOLS.PPG_SYS_DESCEND_MAX_SLP, cycle_cnt) = select_extremum(vpg, range, "min", "min");

        % identify the diastolic peak related points by selecting the [maxima with max amplitude] on vpg as PPG_DIA_IFP
        range = an(ANCOLS.PPG_SYS_DESCEND_MAX_SLP, cycle_cnt):estimated_cycle_end;
        an(ANCOLS.PPG_DIA_IFPM, cycle_cnt) = select_extremum(vpg, range, "max", "max");

        range = an(ANCOLS.PPG_SYS_DESCEND_MAX_SLP, cycle_cnt):an(ANCOLS.PPG_DIA_IFPM, cycle_cnt);
        an(ANCOLS.PPG_DIA_PAM, cycle_cnt) = select_extremum(ppg, range, "min", "min");

        range = an(ANCOLS.PPG_DIA_IFPM, cycle_cnt):estimated_cycle_end;
        an(ANCOLS.PPG_DIA_PEAKM, cycle_cnt) = select_extremum(ppg, range, "max", "max");

        % identify the diastolic peak related points by selecting the [first maxima] on vpg as PPG_DIA_IFP
        range = an(ANCOLS.PPG_SYS_DESCEND_MAX_SLP, cycle_cnt):estimated_cycle_end;
        an(ANCOLS.PPG_DIA_IFPF, cycle_cnt) = select_extremum(vpg, range, "max", "first");

        range = an(ANCOLS.PPG_SYS_DESCEND_MAX_SLP, cycle_cnt):an(ANCOLS.PPG_DIA_IFPF, cycle_cnt);
        an(ANCOLS.PPG_DIA_PAF, cycle_cnt) = select_extremum(ppg, range, "min", "first");

        range = an(ANCOLS.PPG_DIA_IFPF, cycle_cnt):estimated_cycle_end;
        an(ANCOLS.PPG_DIA_PEAKF, cycle_cnt) = select_extremum(ppg, range, "max", "first");

        % identify the end of the cycle
        an(ANCOLS.PPG_CYCLE_END, cycle_cnt) = estimated_cycle_end;

        % extract the features

        if ~isnan(an(ANCOLS.PPG_SYS_PA, cycle_cnt)) && ~isnan(an(ANCOLS.PPG_SYS_ASCEND_MAX_ACC, cycle_cnt))
            fe(FECOLS.IPA1, cycle_cnt) = sum(ppg(an(ANCOLS.PPG_SYS_PA, cycle_cnt) : an(ANCOLS.PPG_SYS_ASCEND_MAX_ACC, cycle_cnt)));
        end
        if ~isnan(an(ANCOLS.PPG_SYS_ASCEND_MAX_ACC, cycle_cnt)) && ~isnan(an(ANCOLS.PPG_SYS_PEAK, cycle_cnt))
            fe(FECOLS.IPA2, cycle_cnt) = sum(ppg(an(ANCOLS.PPG_SYS_ASCEND_MAX_ACC, cycle_cnt) : an(ANCOLS.PPG_SYS_PEAK, cycle_cnt)));
        end

        fe(FECOLS.AIF, cycle_cnt) = (an(ANCOLS.PPG_DIA_IFPF, cycle_cnt) - an(ANCOLS.PPG_SYS_PA, cycle_cnt)) / (an(ANCOLS.PPG_SYS_PEAK, cycle_cnt) - an(ANCOLS.PPG_SYS_PA, cycle_cnt));
        fe(FECOLS.LASIF, cycle_cnt) = an(ANCOLS.PPG_DIA_IFPF, cycle_cnt) - an(ANCOLS.PPG_SYS_PEAK, cycle_cnt);
        
        if ~isnan(an(ANCOLS.PPG_SYS_PEAK, cycle_cnt)) && ~isnan(an(ANCOLS.PPG_DIA_IFPF, cycle_cnt))
            fe(FECOLS.IPAF3, cycle_cnt) = sum(ppg(an(ANCOLS.PPG_SYS_PEAK, cycle_cnt) : an(ANCOLS.PPG_DIA_IFPF, cycle_cnt)));
        end
        if ~isnan(an(ANCOLS.PPG_DIA_IFPF, cycle_cnt)) && ~isnan(an(ANCOLS.PPG_CYCLE_END, cycle_cnt))
            fe(FECOLS.IPAF4, cycle_cnt) = sum(ppg(an(ANCOLS.PPG_DIA_IFPF, cycle_cnt) : an(ANCOLS.PPG_CYCLE_END, cycle_cnt)));
        end

        fe(FECOLS.AIM, cycle_cnt) = (an(ANCOLS.PPG_DIA_IFPM, cycle_cnt) - an(ANCOLS.PPG_SYS_PA, cycle_cnt)) / (an(ANCOLS.PPG_SYS_PEAK, cycle_cnt) - an(ANCOLS.PPG_SYS_PA, cycle_cnt));
        fe(FECOLS.LASIM, cycle_cnt) = an(ANCOLS.PPG_DIA_IFPM, cycle_cnt) - an(ANCOLS.PPG_SYS_PEAK, cycle_cnt);
        
        if ~isnan(an(ANCOLS.PPG_SYS_PEAK, cycle_cnt)) && ~isnan(an(ANCOLS.PPG_DIA_IFPM, cycle_cnt))
            fe(FECOLS.IPAM3, cycle_cnt) = sum(ppg(an(ANCOLS.PPG_SYS_PEAK, cycle_cnt) : an(ANCOLS.PPG_DIA_IFPM, cycle_cnt)));
        end
        if ~isnan(an(ANCOLS.PPG_DIA_IFPM, cycle_cnt)) && ~isnan(an(ANCOLS.PPG_CYCLE_END, cycle_cnt))
            fe(FECOLS.IPAM4, cycle_cnt) = sum(ppg(an(ANCOLS.PPG_DIA_IFPM, cycle_cnt) : an(ANCOLS.PPG_CYCLE_END, cycle_cnt)));
        end
    end

    %%
    % find the end of this ECG cycle (next R-wave)
    if cycle_cnt >= 2 && ~isnan(an(ANCOLS.PPG_SYS_PEAK, cycle_cnt - 1)) && ~isnan(an(ANCOLS.PPG_SYS_PEAK, cycle_cnt))
        fe(FECOLS.PPG_CYCLE, cycle_cnt) = an(ANCOLS.PPG_SYS_PEAK, cycle_cnt) - an(ANCOLS.PPG_SYS_PEAK, cycle_cnt - 1);
        ppg_cycle = update_ppg_cycle(fe(FECOLS.PPG_CYCLE, cycle_cnt), false);
    end

    if cycle_cnt >= 2 && ~isnan(an(ANCOLS.ECG_RWAVE, cycle_cnt - 1)) && ~isnan(an(ANCOLS.ECG_RWAVE, cycle_cnt))
        fe(FECOLS.ECG_CYCLE, cycle_cnt) = an(ANCOLS.ECG_RWAVE, cycle_cnt) - an(ANCOLS.ECG_RWAVE, cycle_cnt - 1);
        ecg_cycle = update_ppg_cycle(fe(FECOLS.ECG_CYCLE, cycle_cnt), false);
        rwave_interval_th = floor(ecg_cycle * 0.5);
    end

    %     figure;
    %     plt_range = min(range) - 50: max(range) + 50;
    %     hold on;
    %     plot(plt_range, ecg_square(plt_range), 'LineWidth', 2, 'Color', '#845EC2');
    %     plot(plt_range, ppg(plt_range), 'LineWidth', 2, 'Color', '#2C73D2');
    %     xline(cycle_start_rwave, 'LineWidth', 2, 'Color', '#B54747');
    %     text(cycle_start_rwave, 1, 'cycle start R-wave', 'Color', '#B54747')
    %     xline(cycle_start_rwave + rwave_interval_th, 'LineWidth', 2, 'Color', '#5A7417');
    %     xline(cycle_start_rwave + rwave_interval_th + ppg_cycle, 'LineWidth', 2, 'Color', '#5A7417');
    %     yline(rwave_amp_th, 'LineWidth', 1.5, 'Color', '#00824B');
    %     xline(cycle_next_rwave, 'LineWidth', 1.5, 'Color', '#B54747');
    %     close;

    cycle_cnt = cycle_cnt + 1;
end
an(:, cycle_cnt:end) = [];
fe(:, cycle_cnt:end) = [];
end

%% estimate the initial value of thresholds
function [rwave_amp_th, rwave_interval_th, ppg_cycle] = estimate_initial_threshold(ecg, ppg, fs, fft_length)
initial_length = fs * 2;

if length(ecg) < initial_length
    start_phase = length(ecg);
else
    start_phase = initial_length;
end

ecg_peak_locs = [];
for p = 2:start_phase-1
    if ecg(p-1) < ecg(p) && ecg(p) > ecg(p+1)
        ecg_peak_locs = [ecg_peak_locs, p];
    end
end

rwave_amp_th = max(ecg(ecg_peak_locs)) / 4;

spectrum = abs(fft(ppg, fft_length));
[~, spectrum_peak_fidx] = max(spectrum(1:fft_length/2));
ppg_cycle_freq = (spectrum_peak_fidx - 1) * (fs / fft_length);
ppg_cycle = fs / ppg_cycle_freq;
rwave_interval_th = ppg_cycle * 0.5;
end


%% get rwave in ecg(range)
function rwave_loc = get_rwave(ecg, range, rwave_amp_th)
rwave_loc = 0;
max_peak = -inf;
for i = range
    if ecg(i-1) < ecg(i) && ecg(i) > ecg(i+1) && ecg(i) > rwave_amp_th && ecg(i) > max_peak
        rwave_loc = i;
        max_peak = ecg(i);
    end
end
end

function rwave_loc = get_rwave_alt(ecg, range, rwave_amp_th)
rwave_loc = 0;
for i = range
    if ecg(i-1) < ecg(i) && ecg(i) > ecg(i+1) && ecg(i) > rwave_amp_th
        rwave_loc = i;
        break;
    end
end

% if rwave_loc ~= 0
%     further_search_limit = floor(rwave_loc + ppg_cycle * 0.5);
%     if further_search_limit >= length(ecg)
%         further_search_limit = length(ecg) - 1;
%     end
%     rwave_loc = select_extremum(ecg, rwave_loc:further_search_limit, "max", "max");
% end

end

%% helper functions
function flag = isminimum(s, ptr)
if s(ptr-1) > s(ptr) && s(ptr) < s(ptr+1)
    flag = true;
else
    flag = false;
end
end

function flag = ismaximum(s, ptr)
if s(ptr-1) < s(ptr) && s(ptr) > s(ptr+1)
    flag = true;
else
    flag = false;
end
end

function flag = return_true(s, ptr)
    flag = true;
end

function flag = larger(val1, val2)
    flag = val1 > val2;
end

function flag = smaller(val1, val2)
    flag = val1 < val2;
end

function [loc, val] = select_extremum(s, range, extremum_type, pooling_type, threshold, threshold_compare)
%% clip the range first 
if max(range) > length(s) - 1
    fprintf("warning: requested range larger than signal length\n");
    range = range(1):(length(s) - 1);   % clip to prevent errors
end

%%
loc = nan;
if isnan(range)
    loc = nan;
    val = nan;
    return;
end

if extremum_type == "max"
    extremum_criterion = @ismaximum;
elseif extremum_type == "min"
    extremum_criterion = @isminimum;
elseif extremum_type == "none"
    extremum_criterion = @return_true;
end

if pooling_type == "max"
    val = -inf;
    pooling_criterion = @larger;
elseif pooling_type == "min"
    val = inf;
    pooling_criterion = @smaller;
elseif pooling_type == "first"
    val = 0;
    pooling_criterion = @return_true;
end

if ~exist("threshold", 'var')
    threshold_criterion = @return_true;
    threshold = 0;
    % set the variable align with the arguments of function "return_true"
else
    if threshold_compare == "larger"
        threshold_criterion = @larger;
    else
        threshold_criterion = @smaller;
    end
end

for ptr = range
    if extremum_criterion(s, ptr) && pooling_criterion(s(ptr), val) && threshold_criterion(s(ptr), threshold)
        loc = ptr;
        val = s(ptr);
        if pooling_type == "first"
            break;
        end
    end
end

if isnan(loc)
    val = nan;
end
end


function estimated_ppg_cycle = update_ppg_cycle(val, clr)
persistent buffer;
persistent ptr;
BUFFER_SIZE = 8;

if isempty(buffer) || clr
    buffer = nan(BUFFER_SIZE, 1);
end
if isempty(ptr) || clr
    ptr = 0;
end

buffer(mod(ptr, BUFFER_SIZE)+1) = val;
estimated_ppg_cycle = median(buffer, 'omitnan');
ptr = ptr + 1;
end

function estimated_ecg_cycle = update_ecg_cycle(val, clr)
persistent buffer;
persistent ptr;
BUFFER_SIZE = 8;

if isempty(buffer) || clr
    buffer = nan(BUFFER_SIZE, 1);
end
if isempty(ptr) || clr
    ptr = 0;
end

buffer(mod(ptr, BUFFER_SIZE)+1) = val;
estimated_ecg_cycle = median(buffer, 'omitnan');
ptr = ptr + 1;
end

function h = get_height(signal, loc)
if ~isnan(loc)
    h = signal(loc);
else
    h = nan;
end
end

