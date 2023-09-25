from dataset import UCILmdbDataset, ICTMatDataset
from utils import AverageMeter

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np

class SampleWeighting():
    def __init__(self, dataset: Dataset) -> None:
        print("init sample weights")
        if type(dataset) == UCILmdbDataset:
            self.w = np.zeros((dataset.__len__(), ))
            self.decay = 0

            case_cnt = 0
            for case_id in dataset.case_list:
                sample_ids = dataset.index_by_case_id[case_id]
            #    sbp_meter = AverageMeter()
            #    dbp_meter = AverageMeter()

            #    for sample_id in sample_ids:
            #        sample = dataset.__getitem__(sample_id)
            #        sbp = np.mean(sample['sbp'])
            #        dbp = np.mean(sample['dbp'])
            #        sbp_meter.update(sbp)
            #        dbp_meter.update(dbp)
                

                sampler = SubsetRandomSampler(sample_ids)
                case_loader = DataLoader(dataset, batch_size=len(sample_ids), sampler=sampler)

                for batch_idx, batch in enumerate(case_loader):
                    dbp = torch.mean(batch['dbp'], dim=-1, keepdim=True).numpy()
                    sbp = torch.mean(batch['sbp'], dim=-1, keepdim=True).numpy()
                    sample_ids = batch['sample-id'].numpy()
                    w = np.abs(sbp - np.mean(sbp)) + np.abs(dbp - np.mean(dbp))
                    w = (w + 1e-4) / (np.max(w) + 1e-4)
            
                    self.w[sample_ids] = np.squeeze(w)
                case_cnt = case_cnt + 1
                if case_cnt % 100 == 0:
                    print(case_cnt, "/", len(dataset.case_list))
    
    def get_weights(self, sample_ids):
        t = max(self.decay - 10, 0)
        w = np.power(self.w[sample_ids], t / 10)
        #w = w / np.sum(w)
        return w


if __name__ == "__main__":
    dataset = UCILmdbDataset("./datasource/ucibpds", split_ratio=[0.7, 0.1, 0.2], load_spectrogram=False, mix_cases_in_trainvalid=False)
    sw = SampleWeighting(dataset)

