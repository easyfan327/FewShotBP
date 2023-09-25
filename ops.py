import colorama
import time

import numpy as np
import torch
from torch.autograd import backward
from torch.utils.data import DataLoader
import torch.nn
import torch._C
from torch.nn.modules.loss import L1Loss
from torch.utils.tensorboard import SummaryWriter

from utils import AverageMeter, PerfStat
from sampleweight import SampleWeighting

def print_status(batch_idx: int, batch_n: int, perf_stat: PerfStat, batch_time: float):
    hours_ETA = ((batch_n - batch_idx) * batch_time) / 3600.0
    print("\r{:6d}/{:6d}".format(batch_idx, batch_n), end=" ")
    print("loss/sbp/dbp[batch]:{:6.2f} {:6.2f} {:6.2f}".format(perf_stat.val["loss"], perf_stat.val["sbp"], perf_stat.val["dbp"]), end=" ")
    print("loss/sbp/dbp[epoch]:{:6.2f} {:6.2f} {:6.2f}".format(perf_stat.avg["loss"], perf_stat.avg["sbp"], perf_stat.avg["dbp"]), end=" ")
    print("Perf:{:3.2f} ETA: {: 4.2f}".format(batch_time, hours_ETA), end=" ")

def train(device: torch.device, dataloader: torch.utils.data.DataLoader, model, optimizer, criterion, perf_stat, tbwriter: SummaryWriter, verbose = True, enable_amp=False, sample_weight: SampleWeighting=None):
    model.train()
    batch_start_time = time.time()
    batch_n = len(dataloader)

    iteration_n = 0

    scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=enable_amp)
    
    for batch_idx, batch in enumerate(dataloader):
        
        # *prepare input data
        data = dict()
        for requested_input_columns in model.requested_input_columns():
            data[requested_input_columns] = batch[requested_input_columns].to(device)

        # *prepare labels
        target = torch.cat((torch.mean(batch['sbp'], dim=-1, keepdim=True), torch.mean(batch['dbp'], dim=-1, keepdim=True)), dim=-1)
        target = target.to(device)

        # *forward propogation
        with torch.cuda.amp.autocast_mode.autocast(enabled=enable_amp):
            output = model(data)
            if sample_weight is not None:
                loss = criterion(output, target, torch.Tensor(sample_weight.get_weights(batch['sample-id'].numpy())).to(device))
            else:
                loss = criterion(output, target)

        # *backward propogation
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()

        # *stat
        perf_stat.update_batch(target.detach().cpu(), output.detach().cpu())
        batch_time = time.time() - batch_start_time

        if verbose:
            print_status(batch_idx, batch_n, perf_stat, batch_time)
        
        if tbwriter is not None:
            tbwriter.add_scalar(tag="perbatch/train/MAE/sbp", scalar_value=perf_stat.val["sbp"], global_step=iteration_n)
            tbwriter.add_scalar(tag="perbatch/train/MAE/dbp", scalar_value=perf_stat.val["dbp"], global_step=iteration_n)
            tbwriter.add_scalar(tag="perbatch/train/loss/abp", scalar_value=perf_stat.val["loss"], global_step=iteration_n)
        
        iteration_n = iteration_n + 1

        batch_start_time = time.time()
    
    if sample_weight is not None:
        sample_weight.decay = sample_weight.decay + 1
    
    return perf_stat


def validate(device: torch.device, dataloader: torch.utils.data.DataLoader, model, criterion, perf_stat, tbwriter, verbose = True):
    model.eval()
    with torch.no_grad():

        batch_start_time = time.time()
        batch_n = len(dataloader)

        iteration_n = 0

        for batch_idx, batch in enumerate(dataloader):

            # *prepare input data
            data = dict()
            for requested_input_columns in model.requested_input_columns():
                data[requested_input_columns] = batch[requested_input_columns].to(device)

            # *prepare labels
            target = torch.cat((torch.mean(batch['sbp'], dim=-1, keepdim=True), torch.mean(batch['dbp'], dim=-1, keepdim=True)), dim=-1)
            target = target.to(device)

            # *forward propogation
            with torch.cuda.amp.autocast_mode.autocast():
                output = model(data)
                loss = criterion(output, target)

            # *stat
            perf_stat.update_batch(target.detach().cpu(), output.detach().cpu())
            batch_time = time.time() - batch_start_time

            if verbose:
                print_status(batch_idx, batch_n, perf_stat, batch_time)
            
            if tbwriter is not None:
                tbwriter.add_scalar(tag="perbatch/valid/MAE/sbp", scalar_value=perf_stat.val["sbp"], global_step=iteration_n)
                tbwriter.add_scalar(tag="perbatch/valid/MAE/dbp", scalar_value=perf_stat.val["dbp"], global_step=iteration_n)
                tbwriter.add_scalar(tag="perbatch/valid/loss/abp", scalar_value=perf_stat.val["loss"], global_step=iteration_n)

            iteration_n = iteration_n + 1

            batch_start_time = time.time()

    return perf_stat

def test(device, dataloader, model):
    model.to(device)
    model.eval()

    outputs = None
    targets = None
    handf_val = None
    handf_std = None
    with torch.no_grad():

        batch_start_time = time.time()
        batch_n = len(dataloader)

        iteration_n = 0

        for batch_idx, batch in enumerate(dataloader):

            # *prepare input data
            data = dict()
            for requested_input_columns in model.requested_input_columns():
                data[requested_input_columns] = batch[requested_input_columns].to(device)

            # *prepare labels
            target = torch.cat((torch.mean(batch['sbp'], dim=-1, keepdim=True), torch.mean(batch['dbp'], dim=-1, keepdim=True)), dim=-1)
            target = target.to(device)

            # *forward propogation
            with torch.cuda.amp.autocast_mode.autocast():
                output = model(data)

            if targets is None:
                targets = target
                outputs = output
                handf_val = batch['handcrafted']
                handf_std = batch['handcrafted-std']
            else:
                targets = torch.concat((targets, target), dim=0)
                outputs = torch.concat((outputs, output), dim=0)
                handf_val = torch.concat((handf_val, batch['handcrafted']))
                handf_std = torch.concat((handf_std, batch['handcrafted-std']))

            batch_time = time.time() - batch_start_time
            iteration_n = iteration_n + 1
            batch_start_time = time.time()

    targets = targets.cpu().numpy()
    outputs = outputs.cpu().numpy()
    handf_val = handf_val.cpu().numpy()
    handf_std = handf_std.cpu().numpy()

    return (targets, outputs, handf_val, handf_std)











