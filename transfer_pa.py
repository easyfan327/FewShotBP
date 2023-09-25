import pickle
import os
import argparse
import pprint
import copy

import numpy as np

import torch
from torch._C import device
import torch.nn
import torch.optim
import torch.cuda
import torch.backends.cudnn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torch.utils.data import Subset

from dataset import UCILmdbDataset, ICTMatDataset
from ops import train, validate, test
from utils import PerfStat, fixseed, PerfStatGroup, dyload_model, correlate
from config import datasets_cfg, models_ft_cfg
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from labelselector import split_trainvalid_clusterbased, split_trainvalid_randomsampling, split_trainvalid_fixed, split_trainvalid_randomsamplingfromfixedrange

from scipy.signal import medfilt
from scipy.stats import linregress


def parseargs():
    parser = argparse.ArgumentParser(description="Transfer Learning using Personalization Adapter")
    parser.add_argument('-m', '--model', default="models.PPGECGNet_V0e2x1b", type=str, help='model name')
    parser.add_argument('-g', '--gpu', default="0", type=str)
    parser.add_argument('-d', '--dataset', default="ucibp")
    parser.add_argument('-e', '--expname', default="ubicomp23a")
    parser.add_argument('-s', '--split_type', default="fixed")
    args = parser.parse_args()
    return args

def get_pearson_rvalue(x, y):
    results = linregress(x, y)
    return results.rvalue

def build_perf_stat_group(name: str) -> PerfStatGroup:
    perf_stat_group = PerfStatGroup(name, metric_funcs={
        "sbp": lambda pd, gt: np.nanmean(np.abs(pd[:, 0] - gt[:, 0])),
        "dbp": lambda pd, gt: np.nanmean(np.abs(pd[:, 1] - gt[:, 1])),
        #"sbp-r": lambda pd, gt: get_pearson_rvalue(pd[:, 0], gt[:, 0]),
        #"dbp-r": lambda pd, gt: get_pearson_rvalue(pd[:, 1], gt[:, 1]),
    })
    return perf_stat_group

def write_epoch_stat_to_tensorboard_hparams(tb_writter: SummaryWriter, params: dict, perf_train: PerfStat, perf_valid: PerfStat, perf_test: PerfStat, epoch: int, tensorboard_savepath: str):
    metrics_dict = dict()
    for perf_stat in (perf_train, perf_valid, perf_test):
        #tb_writter.add_scalars(main_tag=perf_stat.name, tag_scalar_dict=perf_stat.avg, global_step=epoch)
        for metric, val in perf_stat.avg.items():
            metrics_dict["hparams/" + perf_stat.name + "/" + metric] = val
    metrics_dict['hparams/updated_on_epoch'] = epoch
    tb_writter.add_hparams(params, metrics_dict, run_name=os.path.abspath(tensorboard_savepath))

def make_record_path(pretrain_ckpt_path, dataset, expname):
    pretrain_id = os.path.basename(os.path.dirname(pretrain_ckpt_path))
    tsbd_path = os.path.join("./records/", f"pa2{dataset}", f"{pretrain_id}_{expname}.trialdump")
    return tsbd_path

def worker(model, regressor, params):
    hooked = list()
    def fhook(self, layer_input, layer_output):
        hooked.append(layer_output)

    device = torch._C.device("cuda:0")

    #* build dataset
    if params['dataset'] == 'ucibp':
        dataset = UCILmdbDataset(**datasets_cfg['ucibp'])
        txf_cases = dataset.get_percase_samples_for_txflearning(case_min_length=params['min_case'])
    elif params['dataset'] == 'ictdsall':
        dataset = ICTMatDataset(**datasets_cfg['ictdsall'])
        txf_cases = dataset.get_percase_samples_for_txflearning()
    else:
        raise Exception("undefined dataset")

    perf_baseline = build_perf_stat_group("baseline")
    perf_train = build_perf_stat_group("train")
    perf_valid = build_perf_stat_group("valid")
    perf_all = build_perf_stat_group("all")

    for case_counter, case_info in enumerate(txf_cases):
        (case_id, sample_ids) = case_info
        print("{:6d}/{:6d}, ID={:6d}".format(case_counter, len(txf_cases), case_id))

        sample_n = len(sample_ids)
        subset = Subset(dataset, sample_ids)
        dataloader_all = DataLoader(dataset=subset, batch_size=1024, num_workers=8, shuffle=False)

        #! hook the pytorch model to get the intermediate layer output
        #model.dense.register_forward_hook(fhook)
        model.dense[0].register_forward_hook(fhook)
        #model.ppg_feature_gen.temporalblock.tresblock2.pool.register_forward_hook(fhook)
        #model.ecg_feature_gen.temporalblock.tresblock2.pool.register_forward_hook(fhook)
        #! hook the pytorch model to get the intermediate layer output

        #* execute pre-trained model on whole sub-dataset for baseline results
        y_groundtruth, y_nnpredict, af_val, af_std = test(device, dataloader_all, model)
 
        #flattenlayeroutput = torch.flatten(
        #    torch.cat([torch.mean(hooked[0], -1), torch.mean(hooked[1], -1)], -1), 1
        #              ).cpu().numpy()
        flattenlayeroutput = hooked[0].cpu().numpy()
        af_val = KNNImputer(n_neighbors=2, weights='uniform').fit_transform(af_val)
        af_std = KNNImputer(n_neighbors=2, weights='uniform').fit_transform(af_std)

        #* select samples to be labeled
        embeds = None
        clusters = None
        if params['split_type'] == 'fixed':
            train_idx, valid_idx = split_trainvalid_fixed(sample_n, params['train_n'])
        elif params['split_type'] == 'random':
            train_idx, valid_idx = split_trainvalid_randomsampling(sample_n, params['train_n'])
        elif params['split_type'] == 'randomfixedrange':
            train_idx, valid_idx = split_trainvalid_randomsamplingfromfixedrange(sample_n, train_n, params['range_n'])
        elif params['split_type'] == 'cluster':
            train_idx, valid_idx, embeds, clusters = split_trainvalid_clusterbased(sample_n, params['train_n'], af_val, cluster_n = params['train_n'])
        else:
            raise Exception("undefined splitter")

        valid_n = len(valid_idx)
        print(train_idx)

        #* make label
        y = y_groundtruth

        #* make feature
        if params['regressor_input'] == 'nn_af':
            x = np.concatenate((flattenlayeroutput, af_val), axis=1)
        elif params['regressor_input'] == 'nn_afnohr':
            af_val[:, 5:7] = 0
            af_std[:, 5:7] = 0
            x = np.concatenate((flattenlayeroutput, af_val), axis=1)
        elif params['regressor_input'] == 'nn':
            x = flattenlayeroutput
        elif params['regressor_input'] == 'af':
            x = af_val
        else:
            raise Exception("undefined input")

        x = StandardScaler().fit_transform(x)

        #regressor = RandomForestRegressor(**params['regressor_config'])
        regressor.fit(x[train_idx, :], y[train_idx, :])        
        #* make predict
        y_predict = regressor.predict(x)

        perf_baseline.update_batch(gt_batch=y_groundtruth[valid_idx, :], pd_batch=y_nnpredict[valid_idx, :], group=case_id)
        perf_baseline.print_latest_group_stat()
        perf_train.update_batch(gt_batch=y_groundtruth[train_idx, :], pd_batch=y_predict[train_idx, :], group=case_id)
        perf_train.print_latest_group_stat()
        perf_valid.update_batch(gt_batch=y_groundtruth[valid_idx, :], pd_batch=y_predict[valid_idx, :], group=case_id)
        perf_valid.print_latest_group_stat()
        perf_all.update_batch(gt_batch=y_groundtruth, pd_batch=y_predict, group=case_id)

        #! DO NOT FORGET CLEARING THE HOOKED VARS
        hooked = list()
    
    return (perf_baseline, perf_train, perf_valid, perf_all)


if __name__ == "__main__":
    global args
    args = parseargs()

    fixseed(0)

    params = {
        'min_case': 75
    }
    params.update(args.__dict__)
    pprint.pprint(params, indent=4)

    regressor_config = {
        'n_estimators': 20,
        'max_depth': 4,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    }

    model_name = params['model']
    model_path = models_ft_cfg[model_name]['path']

    record_path = make_record_path(pretrain_ckpt_path=model_path, dataset=params['dataset'], expname=params['expname'])
    print(record_path)
    if not os.path.exists(os.path.dirname(record_path)):
        os.makedirs(os.path.dirname(record_path))

    target_model = dyload_model(model_name)
    model = target_model()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    print(params['model'], "loaded")

    trial = list()
    for train_n in list([5, 10, 25, 50]):
        params['train_n'] = train_n
        for rinput in list(['nn_afnohr', 'nn']):
            params['regressor_input'] = rinput
            for n_est in list([20]):
                regressor_config['n_estimators'] = n_est
                for max_depth in list([2]):
                    fixseed(0)
                    regressor_config['max_depth'] = max_depth
                    params['regressor_config'] = regressor_config
                    regressor = RandomForestRegressor(**regressor_config)

                    (perf_baseline, perf_train, perf_valid, perf_all) = worker(model, regressor, params)

                    perf_baseline.remove_lambda_funcs()
                    perf_train.remove_lambda_funcs()
                    perf_valid.remove_lambda_funcs()
                    perf_all.remove_lambda_funcs()
                    
                    trial.append((copy.deepcopy(params), copy.deepcopy(regressor_config), perf_baseline, perf_train, perf_valid, perf_all))
                    pickle.dump(obj=trial, file=open(record_path, "bw"))
