import pprint
import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn

import numpy as np
import matplotlib.pyplot as plt
import random

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
# 
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = float('-inf')
        self.min = float('+inf')
# 
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        elif val < self.min:
            self.min = val
    
    def __repr__(self):
        return "val: {: 8.4f} avg: {: 8.4f}".format(self.val, self.avg)

class BlandAltman(object):
    """Computes axises of Bland-Altman plot"""
    def __init__(self, title):
        self.reset()
        self.title = title
    
    def reset(self):
        self.s1 = np.array([])
        self.s2 = np.array([])

    def add(self, s1_to_add, s2_to_add):
        self.s1 = np.concatenate((self.s1, s1_to_add))
        self.s2 = np.concatenate((self.s2, s2_to_add))
    
    def getfigure(self, alpha=0.5):
        x = np.array(self.s1)
        y = np.array(self.s2)

        fig, ax = plt.subplots(figsize=(12, 6), dpi=144)
        ax.scatter((x+y)/2, x-y, marker=".", linewidths=0.1, alpha=alpha)
        ax.set_xlabel("average")
        ax.set_ylabel("difference")
        ax.set_title(self.title)

    def savefigto(self, path):
        fig = self.getfigure()
        fig.savefig(path)
        fig.close()

class ErrorPlot(object):
    """Computes axises of Error plot"""
    def __init__(self, title, alpha = 1, linewidths = 0.1):
        self.reset()
        self.title = title
        self.alpha = alpha
        self.linewidths = linewidths
    
    def reset(self):
        self.s1 = None
        self.s2 = None

    def add(self, s1_to_add, s2_to_add):
        if self.s1 is None or self.s2 is None:
            self.s1 = s1_to_add
            self.s2 = s2_to_add
        else:
            self.s1 = np.concatenate((self.s1, s1_to_add))
            self.s2 = np.concatenate((self.s2, s2_to_add))
    
    def getfigure(self):
        x = np.array(self.s1)
        y = np.array(self.s2)

        fig, ax = plt.subplots()
        ax.scatter(x, y, marker=".", linewidths = self.linewidths, alpha = self.alpha)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted")
        ax.set_xlim(0, 250)
        ax.set_ylim(0, 250)
        ax.set_title(self.title)
        return fig

    def savefigto(self, path):
        fig = self.getfigure()
        fig.savefig(path)
        fig.close()


class PerfStat():
    def __init__(self, name, metric_funcs: dict) -> None:
        self.name = name
        self.metric_funcs = metric_funcs
        self.count = 0
        self.gt = list()
        self.pd = list()
        self.val = dict()
        self.sum = dict()
        self.avg = dict()

        for metric, func in self.metric_funcs.items():
            self.sum[metric] = 0

    def update_batch(self, gt_batch, pd_batch) -> None:
        batch_size = len(gt_batch)
        self.count += batch_size 
        self.gt.extend(gt_batch)
        self.pd.extend(pd_batch)

        for metric, func in self.metric_funcs.items():
            self.val[metric] = func(gt_batch, pd_batch)
            self.sum[metric] += self.val[metric] * batch_size
            self.avg[metric] = self.sum[metric] / self.count
    
    def remove_lambda_funcs(self):
        self.metric_funcs = None
    
    def avgdict(self):
        # return an dictionary contains items of ("name" + "metric", value)
        out = dict()
        for key, val in self.avg.items():
            out[self.name + "/" + key] = val
        return out

    
class PerfStatGroup(object):
    def __init__(self, name, metric_funcs: dict) -> None:
        self.name = name
        self.metric_funcs = metric_funcs
        self.groups = dict()
        self.groups['global'] = PerfStat("global", self.metric_funcs)
        self.lastest_group = None

    def update_batch(self, gt_batch, pd_batch, group) -> None:
        if group not in self.groups.keys():
            self.groups[group] = PerfStat(group, self.metric_funcs)
        self.groups[group].update_batch(gt_batch, pd_batch)
        self.groups['global'].update_batch(gt_batch, pd_batch)
        self.lastest_group = group
    
    def remove_lambda_funcs(self):
        self.metric_funcs = dict()
        for key, val in self.groups.items():
            val.remove_lambda_funcs()
    
    def print_latest_group_stat(self):
        if self.lastest_group is not None:
            perf_stat = self.groups[self.lastest_group]
            print(self.lastest_group, self.name, ": ", end="")
            pprint.pprint(perf_stat.val)


class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        #print(f"Best: {self.best_loss} Current: {val_loss} Counter: {self.counter}")
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            #print(f"Metric improving")
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            #print(f"Metric not improving")
            self.counter += 1
            # print(f"\nINFO: Early stopping counter {self.counter} of {self.patience}\n")
            if self.counter >= self.patience:
                print('\nINFO: Early stopping\n')
                self.early_stop = True
                
def fixseed(SEED):
    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def signal_normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def signal_normalize_zeromean(data):
    return (data - np.mean(data)) / (np.max(data) - np.min(data))

def correlate(x, y):
    x = signal_normalize_zeromean(x)
    y = signal_normalize_zeromean(y)
    return np.mean(x * y)

def generate_runname(model_name, exp_name):
    exec_timestamp = time.localtime()
    exec_timestr = "{:4d}_{:02d}_{:02d}-{:02d}_{:02d}_{:02d}".format(
        exec_timestamp.tm_year, 
        exec_timestamp.tm_mon, 
        exec_timestamp.tm_mday, 
        exec_timestamp.tm_hour, 
        exec_timestamp.tm_min, 
        exec_timestamp.tm_sec)
    run_name = "{}-{}-{}".format(exp_name, model_name, exec_timestr)
    return run_name

def dyload_model(name) -> torch.nn.Module:
    # model name are specified in format like "models.PPGNet_V0a"
    imported_module = __import__(name)
    class_name = name.split(sep='.')[-1]
    target_model = None

    try:
        print("try loading {:s} from {:s}".format(class_name, name))
        target_model = imported_module.__dict__[class_name].__dict__[class_name]
    except Exception:
        print("failed")
    try:
        print("try loading {:s} from {:s}".format('Model', name))
        target_model = imported_module.__dict__[class_name].__dict__['Model']
    except Exception:
        print("failed")
    
    if issubclass(target_model, torch.nn.Module):
        return target_model
    else:
        raise Exception("failed to load model")

    

FECOLS = list([
"PTT_PA",
"PTT_MAX_ACC",
"PTT_MAX_SLP",
"PTT_MAX_DACC",
"PTT_SYS_PEAK",
"PPG_CYCLE",
"ECG_CYCLE",
"AIF",
"LASIF",
"IPA1",
"IPA2",
"IPAF3",
"IPAF4",
"AIM",
"LASIM",
None,
None,
"IPAM3",
"IPAM4",
"SDPTG_DA",
"SDPTG_BA",
"AGI"
])