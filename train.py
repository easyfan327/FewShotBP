import time
import os
import random
import threading
import argparse
import json
import pprint

import numpy as np

import torch
from torch._C import device
import torch.nn
import torch.optim
import torch.cuda
import torch.backends.cudnn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset import UCILmdbDataset
from ops import train, validate
from utils import EarlyStopping, PerfStat, fixseed, generate_runname
from sampleweight import SampleWeighting

def parseargs():
    parser = argparse.ArgumentParser(description="PPG Based Blood Pressure NN Model on UCI PPG/ECG Dataset")
    parser.add_argument('-r', '--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('-d', '--dataset', default='./datasource/ucibpds', type=str, help='path to dataset')
    parser.add_argument('-e', '--expname', default='ubicomp23a', type=str, help='experiment name')
    parser.add_argument('-m', '--model', default="models.PPGECGNet_V0e2x1b", type=str, help='model name')
    parser.add_argument('-p', '--hyperparams', default='./hyperparams/Pretrain-BPCRNN.json', type=str, help='path to a json file that contains hyperparameters')
    parser.add_argument('-g', '--gpu', default="0", type=str)
    parser.add_argument('-w', '--sw', default="off", type=str)
    args = parser.parse_args()
    return args

def write_epoch_stat_to_tensorboard(tb_writter: SummaryWriter, perf_train: PerfStat, perf_valid: PerfStat, perf_test: PerfStat, epoch: int):
    for perf_stat in (perf_train, perf_valid, perf_test):
        #tb_writter.add_scalars(main_tag=perf_stat.name, tag_scalar_dict=perf_stat.avg, global_step=epoch)
        for metric, val in perf_stat.avg.items():
            tb_writter.add_scalar(tag=perf_stat.name + "/" + metric, scalar_value=val, global_step=epoch)

def write_epoch_stat_to_tensorboard_hparams(tb_writter: SummaryWriter, params: dict, perf_train: PerfStat, perf_valid: PerfStat, perf_test: PerfStat, epoch: int, tensorboard_savepath: str):
    metrics_dict = dict()
    for perf_stat in (perf_train, perf_valid, perf_test):
        #tb_writter.add_scalars(main_tag=perf_stat.name, tag_scalar_dict=perf_stat.avg, global_step=epoch)
        for metric, val in perf_stat.avg.items():
            metrics_dict["hparams/" + perf_stat.name + "/" + metric] = val
    metrics_dict['hparams/updated_on_epoch'] = epoch
    tb_writter.add_hparams(params, metrics_dict, run_name=os.path.abspath(tensorboard_savepath))

def save_status(model, optimizer, lr_scheduler, params, perf_train: PerfStat, perf_valid: PerfStat, perf_test: PerfStat, epoch, checkpoint_savepath):
    #* save current best epoch
    perf_train.remove_lambda_funcs()
    perf_valid.remove_lambda_funcs()
    perf_test.remove_lambda_funcs()
    #! remove lambda functions or torch.save will crash
    to_save = dict()
    to_save['model'] = model.state_dict()
    to_save['optimizer'] = optimizer.state_dict()
    to_save['lr_scheduler'] = lr_scheduler.state_dict()
    to_save['updated_on_epoch'] = epoch
    to_save['perf_train'] = perf_train
    to_save['perf_valid'] = perf_valid
    to_save['perf_test'] = perf_test
    to_save['hp'] = params

    torch.save(to_save, os.path.join(checkpoint_savepath, "ckpt-best.pth"))
    json.dump(params, open(os.path.join(checkpoint_savepath, "hp.json"), "w"))
    print("checkpoint saved in {:s}".format(checkpoint_savepath))

def build_perf_stat(name: str, criterion) -> PerfStat:
    perf_stat = PerfStat(name, metric_funcs={
        "sbp": lambda pd, gt: torch.nn.L1Loss()(pd[:, 0], gt[:, 0]),
        "dbp": lambda pd, gt: torch.nn.L1Loss()(pd[:, 1], gt[:, 1]),
        "loss": lambda pd, gt: criterion(pd, gt)
    })
    return perf_stat

if __name__ == "__main__":
    global args
    args = parseargs()

    fixseed(0)
    #* load the model class dynamically
    imported_module = __import__(args.model)
    class_name = args.model.split(sep='.')[-1]
    target_model = imported_module.__dict__[class_name].__dict__[class_name]

    #* select the gpu to be usesd
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch._C.device("cuda:0")

    #* setup paths
    run_name = generate_runname(model_name=target_model.__name__, exp_name=args.expname)
    checkpoint_savepath = os.path.join("./checkpoints/", args.expname, run_name)
    tensorboard_savepath = os.path.join("./tensorboard/", args.expname, run_name)
    summary_writer = SummaryWriter(tensorboard_savepath)
    if not os.path.exists(checkpoint_savepath):
        os.makedirs(checkpoint_savepath)

    #* load hyper-parameters
    params = dict()
    hyperparams_path = args.hyperparams
    with open(hyperparams_path, 'r') as f:
        params = json.load(f)
    print(args.dataset)
    print(checkpoint_savepath)
    print(tensorboard_savepath)

    #* build model/ optimizer
    model = target_model()
    model.to(device)

    criterion = torch.nn.SmoothL1Loss(reduction='mean', beta=params["smoothl1loss_beta"]).to(device)
    #def criterion(ypred, ygt, weight = None):
    #    if weight is None:
    #        loss = torch.mean(torch.square((ypred - ygt)))
    #    else:
    #        loss = torch.mean(weight.unsqueeze(-1).repeat(1,2) * torch.square((ypred - ygt)))
    #    return loss

    optimizer = torch.optim.RMSprop(model.parameters(), lr=params["lr"], weight_decay=params["l2norm"])

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, 
        milestones=tuple([int(s) for s in params["lrsched_step"].split(sep=",")]), 
        gamma=params["lrsched_gamma"])
    es = EarlyStopping(patience=params['es_patience'], min_delta=params['es_min_delta'])
    
    if args.resume:
        print("loading checkpoint {}".format(args.resume))
        checkpoint = torch.load(args.resume)
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print("checkpoint {} loaded, last epoch {}".format(args.resume, last_epoch))
    else:
        last_epoch = -1

    #* build dataset/ dataset sampler
    dataset = UCILmdbDataset(lmdb_folder=args.dataset, load_spectrogram=False, split_ratio=list([float(s) for s in params["train_valid_test_split_ratio"].split(sep=",")]), mix_cases_in_trainvalid=params['mixed_cases_in_trainvalid'])
    (train_sampler, valid_sampler) = dataset.get_trainvalidsampler(fold=0)
    test_sampler = dataset.get_testsampler(case_min_length=params['test_case_min_length'])

    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=params["batchsize"], num_workers=params["loader_worker"], pin_memory=True)
    valid_dataloader = DataLoader(dataset, sampler=valid_sampler, batch_size=params["batchsize"], num_workers=params["loader_worker"], pin_memory=True)
    test_dataloader = DataLoader(dataset, sampler=test_sampler, batch_size=params["batchsize"], num_workers=params["loader_worker"], pin_memory=True)

    #* records training meta information
    params["criterion"] = criterion.__class__.__name__
    params["optimizer"] = optimizer.__class__.__name__
    params["lr_scheduler"] = lr_scheduler.__class__.__name__

    global_start_time = time.time()

    best_metric = float("+inf")

    #* samples re-weighting
    params['sw'] = args.sw
    if args.sw == "on":
        sw = SampleWeighting(dataset)
    else:
        sw = None
    
    print("hyperparameters for the run:")
    pprint.pprint(params, width=1)

    for epoch in range(last_epoch + 1, params["max_epoch"]):

        #* train
        print("\nEpoch {:3d}: Training".format(epoch))
        perf_train = build_perf_stat("train", criterion)
        perf_train = train(device=device, dataloader=train_dataloader, model=model, optimizer=optimizer, criterion=criterion, perf_stat=perf_train, tbwriter=summary_writer, enable_amp=params["enable_amp"], sample_weight=sw)

        #print("sw decay", sw.decay)

        #* validate
        print("\nEpoch {:3d}: Validating".format(epoch))
        perf_valid = build_perf_stat("valid", criterion)
        perf_valid = validate(device=device, dataloader=valid_dataloader, model=model, criterion=criterion, perf_stat=perf_valid, tbwriter=None)

        #* test
        print("\nEpoch {:3d}: Testing".format(epoch))
        perf_test = build_perf_stat("test", criterion)
        perf_test = validate(device=device, dataloader=test_dataloader, model=model, criterion=criterion, perf_stat=perf_test, tbwriter=None)
        lr_scheduler.step()

        write_epoch_stat_to_tensorboard(summary_writer, perf_train, perf_valid, perf_test, epoch)

        if perf_valid.avg["loss"] < best_metric:
            current_best_epoch = epoch
            t = threading.Thread(target=save_status, args=(model, optimizer, lr_scheduler, params, perf_train, perf_valid, perf_test, epoch, checkpoint_savepath))
            t.start()
            #save_status(model, optimizer, lr_scheduler, params, perf_train, perf_valid, perf_test, epoch, checkpoint_savepath)
            write_epoch_stat_to_tensorboard_hparams(summary_writer, params, perf_train, perf_valid, perf_test, epoch, tensorboard_savepath)
            best_metric = perf_valid.avg["loss"]
        
        es(perf_valid.avg["loss"])
        if es.early_stop and params['es_enable']:
            break
        
    write_epoch_stat_to_tensorboard_hparams(summary_writer, params, perf_train, perf_valid, perf_test, epoch, tensorboard_savepath)
    
        
