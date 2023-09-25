import time
import os
import argparse
import pprint
import copy
import pickle

import numpy as np

import torch
from torch._C import device
import torch.nn
from torch.nn.functional import smooth_l1_loss
import torch.optim
import torch.cuda
import torch.backends.cudnn

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler

from dataset import UCILmdbDataset, ICTMatDataset
from ops import train, validate
from utils import EarlyStopping, PerfStat, fixseed, dyload_model, PerfStatGroup
from config import datasets_cfg, models_ft_cfg
from labelselector import split_trainvalid_fixed, split_trainvalid_randomsampling

def parseargs():
    parser = argparse.ArgumentParser(description="PPG Based Blood Pressure NN Model on UCI PPG/ECG Dataset")
    parser.add_argument('-m', '--model', default="models.PPGECGNet_V0e2x1b", type=str, help='model name')
    parser.add_argument('-g', '--gpu', default="0", type=str)
    parser.add_argument('-t', '--tune', default="v0", type=str)
    parser.add_argument('-n', '--trainn', default=5)
    parser.add_argument('-d', '--dataset', default="ucibp")
    parser.add_argument('-s', '--split_type', default="fixed", type=str, help='split type')
    parser.add_argument('-e', '--expname', default='UbiComp23a', type=str, help='experiment name')
    args = parser.parse_args()
    return args

def build_perf_stat(name: str, criterion) -> PerfStat:
    perf_stat = PerfStat(name, metric_funcs={
        "sbp": lambda pd, gt: torch.nn.L1Loss()(pd[:, 0], gt[:, 0]),
        "dbp": lambda pd, gt: torch.nn.L1Loss()(pd[:, 1], gt[:, 1]),
        "loss": lambda pd, gt: criterion(pd, gt)
    })
    return perf_stat

def make_tsbd_path(pretrain_ckpt_path, expname, dataset, tune, trainn):
    pretrain_id = os.path.basename(os.path.dirname(pretrain_ckpt_path))
    tsbd_path = os.path.join("./tensorboard/", f"transfer2{dataset}-{expname}", f"{pretrain_id}_{tune}_{trainn}")
    return tsbd_path

def write_to_tensorboard_hparams(tb_writter: SummaryWriter, params: dict, perf_train: PerfStatGroup, perf_valid: PerfStatGroup, tensorboard_savepath: str):
    metrics_dict = dict()
    for perf_stat in (perf_train, perf_valid):
        for metric, val in perf_stat.groups['global'].avg.items():
            metrics_dict["hparams/" + perf_stat.name + "/" + metric] = val
    tb_writter.add_hparams(params, metrics_dict, run_name=os.path.abspath(tensorboard_savepath))

if __name__ == "__main__":
    global args
    args = parseargs()
    params ={
        'batchsize': 256,
        'loader_worker': 4,
        'smoothl1loss_beta': 5,
        'lr': 1e-5,
        'l2norm': 1e-4,
        'enable_amp': False,
        "es_patience": 3,
        "es_min_delta": 0.01,
        "es_enable": True,
        'min_case': 75,
        'trainn': 10
    }

    fixseed(0)

    #* select the gpu to be usesd
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch._C.device("cuda:0")

    params.update(args.__dict__)
    pprint.pprint(params)
    
    target_model = dyload_model(params['model'])
    ckpt_path = models_ft_cfg[params['model']]['path']
    checkpoint = torch.load(ckpt_path)

    tensorboard_path = make_tsbd_path(ckpt_path, params['expname'], params['dataset'], params['tune'], params['trainn'])
    print(tensorboard_path)
    summary_writer = SummaryWriter(tensorboard_path)

    if params['dataset'] == 'ucibp':
        dataset = UCILmdbDataset(**datasets_cfg['ucibp'])
        txf_cases = dataset.get_percase_samples_for_txflearning(case_min_length=params['min_case'])
    elif params['dataset'] == 'ictdsall':
        dataset = ICTMatDataset(**datasets_cfg['ictdsall'])
        txf_cases = dataset.get_percase_samples_for_txflearning()
    else:
        raise Exception("undefined dataset")

    train_n = int(params['trainn'])

    perf_train_stat_group = PerfStatGroup(name="train_perf_group", metric_funcs={
        "sbp": lambda pd, gt: torch.nn.L1Loss()(pd[:, 0], gt[:, 0]),
        "dbp": lambda pd, gt: torch.nn.L1Loss()(pd[:, 1], gt[:, 1]),
    })
    perf_valid_stat_group = PerfStatGroup(name="valid_perf_group", metric_funcs={
        "sbp": lambda pd, gt: torch.nn.L1Loss()(pd[:, 0], gt[:, 0]),
        "dbp": lambda pd, gt: torch.nn.L1Loss()(pd[:, 1], gt[:, 1]),
    })

    for case_counter, case_info in enumerate(txf_cases):
        (case_id, sample_ids) = case_info
        print(f"{case_counter}/{len(txf_cases)}, personalizing case {case_id}")

        sample_n = len(sample_ids)

        if params['split_type'] == 'fixed':
            train_idx, valid_idx = split_trainvalid_fixed(sample_n, train_n)
        elif params['split_type'] == 'random':
            train_idx, valid_idx = split_trainvalid_randomsampling(sample_n, train_n)
        else:
            raise Exception("Undefined Split")
        
        target_train_id = [sample_ids[i] for i in train_idx]
        target_valid_id = [sample_ids[i] for i in valid_idx]
        
        train_sampler = SubsetRandomSampler(target_train_id)
        valid_sampler = SubsetRandomSampler(target_valid_id)

        train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=params["batchsize"], num_workers=params["loader_worker"])
        valid_dataloader = DataLoader(dataset, sampler=valid_sampler, batch_size=params["batchsize"], num_workers=params["loader_worker"])

        '''
        Prepare training 
        '''
        model = target_model()
        model.to(device=device)
        model.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.RMSprop(model.parameters(), lr=params["lr"], weight_decay=params["l2norm"])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        es = EarlyStopping(patience=params['es_patience'], min_delta=params['es_min_delta'])
        criterion = torch.nn.SmoothL1Loss(reduction='mean', beta=params["smoothl1loss_beta"]).to(device) 
        def cri(out, tgt):
            loss = 0.35 * smooth_l1_loss(out[:, 0], tgt[:, 0]) + 0.65 * smooth_l1_loss(out[:, 1], tgt[:, 1])
            return  loss
        criterion = cri
        

        #* pretrain evaluation
        perf_pretrain = build_perf_stat(name="pretrain", criterion=criterion)
        perf_pretrain = validate(device=device, dataloader=valid_dataloader, model=model, criterion=criterion, perf_stat=perf_pretrain, tbwriter=None, verbose=False)

        #! set tunable layers here
        model.set_tunable_layers(params['tune'])
        
        t0 = time.time()
        best_perf_valid = copy.deepcopy(perf_pretrain)
        best_perf_train = None
        for epoch in range(200):
            #* train
            print("\nEpoch {:3d}: Training".format(epoch))
            if train_n > 0:
                perf_train = build_perf_stat(name="train", criterion=criterion)
                perf_train = train(device=device, dataloader=train_dataloader, model=model, optimizer=optimizer, criterion=criterion, perf_stat=perf_train, tbwriter=None, enable_amp=params["enable_amp"], verbose=False)
            pprint.pprint(perf_train.avgdict())

            #* validate
            print("\nEpoch {:3d}: Validating".format(epoch))
            perf_valid = build_perf_stat(name="valid", criterion=criterion)
            perf_valid = validate(device=device, dataloader=valid_dataloader, model=model, criterion=criterion, perf_stat=perf_valid, tbwriter=None, verbose=False)
            pprint.pprint(perf_valid.avgdict())

            #sched.step(perf_valid.avg['loss'])

            if perf_valid.avg["loss"] < best_perf_valid.avg["loss"]:

                #* update current best performance
                best_epoch = epoch
                best_perf_train = copy.deepcopy(perf_train)
                best_perf_valid = copy.deepcopy(perf_valid)
 
            es(perf_valid.avg["loss"])
            if es.early_stop and params['es_enable']:
                break

        for key, value in best_perf_valid.avg.items():
            summary_writer.add_scalar(tag=key, scalar_value=value, global_step=case_counter)
            summary_writer.add_scalar(tag="diff/" + key, scalar_value=value - perf_pretrain.avg[key], global_step=case_counter)

        if best_perf_train is not None: 
            perf_train_stat_group.update_batch(torch.stack(best_perf_train.gt), torch.stack(best_perf_train.pd), group=case_id)
        perf_valid_stat_group.update_batch(torch.stack(best_perf_valid.gt), torch.stack(best_perf_valid.pd), group=case_id)

        for key, value in perf_valid_stat_group.groups['global'].avg.items():
            summary_writer.add_scalar(tag="avg_progress/" + key, scalar_value=value, global_step=case_counter)
                
        print("Case", case_id, "trained with", epoch, "epoches")
        print("Loss/SBP/DBP: {:6.2f} {:6.2f} {:6.2f}".format(perf_pretrain.avg["loss"], perf_pretrain.avg["sbp"], perf_pretrain.avg["dbp"]))
        print("->")
        print("Loss/SBP/DBP: {:6.2f} {:6.2f} {:6.2f}".format(best_perf_valid.avg["loss"], best_perf_valid.avg["sbp"], best_perf_valid.avg["dbp"]))

    write_to_tensorboard_hparams(summary_writer, params, perf_train_stat_group, perf_valid_stat_group, tensorboard_path)

    perf_train_stat_group.remove_lambda_funcs()
    perf_valid_stat_group.remove_lambda_funcs()
    pickle.dump(perf_train_stat_group, open(os.path.join(tensorboard_path, "perf_train_stat_group") + ".trialdump", "wb"))
    pickle.dump(perf_valid_stat_group, open(os.path.join(tensorboard_path, "perf_valid_stat_group") + ".trialdump", "wb"))

        
        
