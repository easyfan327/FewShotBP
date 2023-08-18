function set_globals()
%get_globals setting all the global variables for both app and batch
%processing algorithm
%   此处显示详细说明
global ANCOLS FECOLS CYCLE_MAX_N ANNO_N FENO_N PAT_MIN FS

ANNO_N = 30;
FENO_N = 25;

CYCLE_MAX_N = 5000;

%% annotation columns - assumed to be constant
ANCOLS.ECG_RWAVE                    = 1;
ANCOLS.PPG_SYS_PA                   = 2;
ANCOLS.PPG_SYS_ASCEND_MAX_ACC       = 3;        % aka. SDPTG Point a
ANCOLS.SDPTG_A                      = 3;
ANCOLS.PPG_SYS_ASCEND_MAX_SLP       = 4;
ANCOLS.PPG_SYS_ASCEND_MAX_DACC      = 5;        % aka. SDPTG Point b
ANCOLS.SDPTG_B                      = 5;
ANCOLS.PPG_SYS_PEAK                 = 6;
ANCOLS.PPG_SYS_DESCEND_MAX_SLP      = 7;
ANCOLS.PPG_DIA_PAM                  = 10;
ANCOLS.PPG_DIA_IFPM                 = 11;
ANCOLS.PPG_DIA_PEAKM                = 12;
ANCOLS.PPG_DIA_PAF                  = 13;
ANCOLS.PPG_DIA_IFPF                 = 14;
ANCOLS.PPG_DIA_PEAKF                = 15;
ANCOLS.PPG_CYCLE_END                = 20;

ANCOLS.DBG_SYS_PEAK_FIND_START      = 21;
ANCOLS.DBG_SYS_PEAK_FIND_END        = 22;
ANCOLS.DBG_RWAVE_FIND_START     = 23;
ANCOLS.DBG_RWAVE_FIND_END       = 24;

ANCOLS.SDPTG_C                      = 25;
ANCOLS.SDPTG_D                      = 26;
ANCOLS.SDPTG_E                      = 27;

%% feature columns - assumed to be constant
FECOLS.PTT_PA                       = 1;
FECOLS.PTT_MAX_ACC                  = 2;
FECOLS.PTT_MAX_SLP                  = 3;
FECOLS.PTT_MAX_DACC                 = 4;
FECOLS.PTT_SYS_PEAK                 = 5;
FECOLS.PPG_CYCLE                    = 6;     %systolic peak - systolic peak
FECOLS.ECG_CYCLE                    = 7;     %RR-interval
FECOLS.AIF                          = 8;     %augmentation index
FECOLS.LASIF                        = 9;     %large artery stiffness index
FECOLS.IPA1                         = 10;    %inflection point area ratio 1
FECOLS.IPA2                         = 11;    %inflection point area ratio 2
FECOLS.IPAF3                        = 12;    %inflection point area ratio 3
FECOLS.IPAF4                        = 13;    %inflection point area ratio 4
FECOLS.AIM                          = 14;    %augmentation index
FECOLS.LASIM                        = 15;    %large artery stiffness index
FECOLS.IPAM3                        = 18;    %inflection point area ratio
FECOLS.IPAM4                        = 19;    %inflection point area ratio

FECOLS.SDPTG_DA                     = 20;
FECOLS.SDPTG_BA                     = 21;
FECOLS.AGI                          = 22;    %aging index

FS = 125;

PAT_MIN = floor(0.1 * FS);  % minimum PAT is 100ms

% PPG_SIG_CHANNEL = 1;
% ABP_SIG_CHANNEL = 2;
% PPG_DIFF1 = 3;
% PPG_DIFF2 = 4;
%
% PPG_ENV_STD_TH = 0.2;
% ABP_ENV_STD_TH = 5;
%
% FS = 125;
% WINDOW_LEN = 5;
% WINDOW_OVERLAP = 3;
% FRAME_LEN = WINDOW_LEN * FS;
% OVERLAP_LEN = WINDOW_OVERLAP * FS;
end

