import pickle
import pprint
import numpy as np
import random
import sys
import os

import lmdb
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import KFold

from utils import fixseed
from scipy.io import loadmat

# code for reference only
'''
txn = lmdbenv.begin(write=True)
txn.put(key="index_by_sample_id".encode(),
        value=pickle.dumps(index_by_sample_id))
txn.put(key="index_by_case_id".encode(), value=pickle.dumps(index_by_case_id))
txn.put(key="case_list".encode(), value=pickle.dumps(case_list))
txn.commit()
'''
# code for reference only

LMDB_MAP_SIZE = 1000 * 1000 * 1000 * 1000 # 1T

class ICTMatDataset(Dataset):
    def __init__(self, dataset_folder, subjects, exps=range(1,11)):
        super(ICTMatDataset, self).__init__()
        self.dataset_folder = dataset_folder
        self.samples = list()
        self.subjects = subjects
        self.subject_sampleindex = dict()
        sample_cnt = 0

        for subjectno in subjects:
            self.subject_sampleindex[subjectno] = list()
            for expgroupno in range(1, 4):
                for expno in exps:
                    expected_filename = "{:02d}-{:02d}-{:02d}.mat".format(subjectno, expgroupno, expno)
                    matfile = None
                    try:
                        matfile = loadmat(os.path.join(dataset_folder, expected_filename))
                        self.samples.append(matfile)
                        self.subject_sampleindex[subjectno].append(sample_cnt)
                        sample_cnt = sample_cnt + 1
                    except Exception:
                        print("exception on exp {:02d}-{:02d}-{:02d}".format(subjectno, expgroupno, expno))
        
        
        self.fs = 125
        self.ppgstd = 0.550721
        self.ecgstd = 0.156662
        #PPG mean 0.000040, std 0.550721
        #ECG mean -0.000000, std 0.156662

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample = dict()

        sample['ppg'] = self._normalize(np.squeeze(self.samples[index]['ppg']).astype(dtype=np.float32), self.ppgstd)
        sample['ecg'] = self._normalize(np.squeeze(self.samples[index]['ecg']).astype(dtype=np.float32), self.ecgstd)
        sample['sbp'] = np.expand_dims(np.squeeze(self.samples[index]['exp_sbp']), axis=0).astype(dtype=np.float32)
        sample['dbp'] = np.expand_dims(np.squeeze(self.samples[index]['exp_dbp']), axis=0).astype(dtype=np.float32)
        sample['handcrafted'] = np.squeeze(self.samples[index]['fval']).astype(dtype=np.float32)
        sample['handcrafted-std'] = np.squeeze(self.samples[index]['fstd']).astype(dtype=np.float32)
        
        sample['sig'] = np.concatenate((np.expand_dims(sample['ppg'], axis=-1), np.expand_dims(sample['ecg'], axis=-1)), axis=-1)

        exp_type = self.samples[index]['exp_type']
        if len(exp_type) == 0:
            exp_type_embed = np.asarray([0])
        elif exp_type[0] == 'ice':
            exp_type_embed = np.asarray([1])
        elif exp_type[0] == 'stair':
            exp_type_embed = np.asarray([2])
        elif exp_type[0] == 'walk':
            exp_type_embed = np.asarray([3])
        elif exp_type[0] == 'hiit':
            exp_type_embed = np.asarray([4])
        elif exp_type[0] == 'hit':
            exp_type_embed = np.asarray([4])
        else:
            exp_type_embed = np.asarray([5])
        tmp = np.zeros((5, ))
        tmp[exp_type_embed[0]] = 1
        #print(tmp)
        sample['handcrafted'] = np.concatenate((tmp, sample['handcrafted']))
        # !make sure the ndarray is writeable and owns its own data
        for k in sample:
            sample[k] = np.require(sample[k], requirements=['O', 'W'])
            sample[k].setflags(write=1)

        return sample
    
    def _normalize(self, x, target_std):
        x = (x * target_std) / np.std(x) 
        return x
    
    def data_columns(self)->list:
        return ['ppg', 'abp', 'sbp', 'dbp', 'ecg', 'cwtppg', 'cwtecg', 'sig', 'handcrafted', 'handcrafted-std']

    def get_percase_samples_for_txflearning(self):
        print("{:s} {:s}".format(self.__class__.__name__, sys._getframe().f_code.co_name))
        
        out_list = list()
        for case_id in self.subjects:
            try:
                sample_ids = self.subject_sampleindex[case_id]
                out_list.append((case_id, sample_ids))
            except:
                print("exception on case", case_id)
        
        return out_list

class UCILmdbDataset(Dataset):
    def __init__(self, lmdb_folder, load_spectrogram=False, split_ratio=[0.72, 0.18, 0.1], mix_cases_in_trainvalid=True):
        super(UCILmdbDataset, self).__init__()
        self.dataset_folder = lmdb_folder
        self.lmdbenv = lmdb.open(lmdb_folder, map_size=LMDB_MAP_SIZE)
        self.lmdbtxn = self.lmdbenv.begin()

        self.load_spectrogram = load_spectrogram

        self.case_list:list = pickle.loads(self.lmdbtxn.get("case_list".encode()))
        self.index_by_case_id:dict = pickle.loads(self.lmdbtxn.get("index_by_case_id".encode()))
        self.index_by_sample_id = pickle.loads(self.lmdbtxn.get("index_by_sample_id".encode()))

        self.total_case_n = len(self.case_list)
        self.total_sample_n = len(self.index_by_sample_id)
        self.split_ratio = split_ratio
        self.kfold = int((self.split_ratio[0] + self.split_ratio[1]) / self.split_ratio[1])

        self.trainvalid_case_n = int(self.total_case_n * (self.split_ratio[0] + self.split_ratio[1]))
        self.cases_for_trainvalid = self.case_list[0 : self.trainvalid_case_n]
        self.cases_for_test = self.case_list[self.trainvalid_case_n : ]
        self.mix_cases_in_trainvalid = mix_cases_in_trainvalid

        kf = KFold(self.kfold)

        if mix_cases_in_trainvalid:
            print("train/valid samples are sampled from the same cases set")
            sample_ids_for_trainvalid = self.get_samples_from_case_list(self.cases_for_trainvalid)
            trainvalid_sample_n = len(sample_ids_for_trainvalid)
            #! shuffling is important, otherwise the partition will approximate the condition when mix_cases = false

            random.shuffle(sample_ids_for_trainvalid)
            self.trainvalid_sample_folds = [
                (
                    self.slicing_lista_by_listb(sample_ids_for_trainvalid, samples_for_train), 
                    self.slicing_lista_by_listb(sample_ids_for_trainvalid, samples_for_valid)
                )
                for samples_for_train, samples_for_valid in kf.split(range(trainvalid_sample_n))]
        else:
            print("train/valid samples are sampled from different cases sets")
            self.trainvalid_case_folds = [
                (
                    self.slicing_lista_by_listb(self.case_list, cases_for_train), 
                    self.slicing_lista_by_listb(self.case_list, cases_for_valid)
                ) 
                for cases_for_train, cases_for_valid in kf.split(range(self.trainvalid_case_n))]

        print("{:s} initialized with following configuration:".format(self.__class__.__name__))
        pprint.pprint(
        {
            "Cases": 
            {
                "Total": len(self.case_list),
                "Train/Valid": len(self.cases_for_trainvalid),
                "Test": len(self.cases_for_test)
            },
            "Samples": self.total_sample_n, 
            "Split Ratio": self.split_ratio, 
            "Folds": self.kfold
        }, width=1
        )

    def get_samples_from_case_list(self, case_list: list) -> list:
        sample_ids = list()
        for case in case_list:
            sample_ids.extend(self.index_by_case_id[case])
        return sample_ids
    
    def get_samples_from_case_id(self, case_id) -> list:
        return self.index_by_case_id[case_id]
    
    def get_trainvalidsampler(self, fold=0) -> tuple:
        print("{:s} {:s}".format(self.__class__.__name__, sys._getframe().f_code.co_name))

        if self.mix_cases_in_trainvalid:
            sample_ids_for_train = self.trainvalid_sample_folds[fold][0]
            sample_ids_for_valid = self.trainvalid_sample_folds[fold][1]

            pprint.pprint(
                {
                    "Samples":
                    {
                        "Train": len(self.trainvalid_sample_folds[fold][0]),
                        "Valid": len(self.trainvalid_sample_folds[fold][1])
                    }
                }
            )

        else:
            sample_ids_for_train = self.get_samples_from_case_list(self.trainvalid_case_folds[fold][0])
            sample_ids_for_valid = self.get_samples_from_case_list(self.trainvalid_case_folds[fold][1])

            pprint.pprint(
                {
                    "Samples":
                    {
                        "Train": len(sample_ids_for_train),
                        "Valid": len(sample_ids_for_valid)
                    },
                    "Cases":
                    {
                        "Train": len(self.trainvalid_case_folds[fold][0]),
                        "Valid": len(self.trainvalid_case_folds[fold][1])
                    }
                }
            )

        print("random sampler generated")
        return (SubsetRandomSampler(sample_ids_for_train), SubsetRandomSampler(sample_ids_for_valid))    
    
    def get_testsampler(self, case_min_length) -> SubsetRandomSampler:
        print("{:s} {:s}".format(self.__class__.__name__, sys._getframe().f_code.co_name))

        out_case_list = list()
        for case_id in self.cases_for_test:
            try:
                sample_id = self.index_by_case_id[case_id]
                if len(sample_id) > case_min_length:
                    out_case_list.append(case_id)
            except:
                print("exception on case", case_id)
        sample_ids_for_test = self.get_samples_from_case_list(out_case_list)
        pprint.pprint(
            {
                "Samples":
                {
                    "Test": len(sample_ids_for_test),
                },
                "Cases":
                {
                    "Test": len(out_case_list),
                }
            }
        )

        return SubsetRandomSampler(sample_ids_for_test)
    
    def get_percase_samples_for_txflearning(self, case_min_length):
        print("{:s} {:s}".format(self.__class__.__name__, sys._getframe().f_code.co_name))
        
        out_list = list()
        for case_id in self.cases_for_test:
            try:
                sample_ids = self.index_by_case_id[case_id]
                if len(sample_ids) > case_min_length:
                    out_list.append((case_id, sample_ids))
            except:
                print("exception on case", case_id)
        
        return out_list
            
    def before_pickle(self):
        self.lmdbenv = None
        self.lmdbtxn = None

    def slicing_lista_by_listb(self, lista: list, listb: list) -> list:
        return [lista[idx] for idx in listb]

    def __len__(self):
        return len(self.index_by_sample_id)
    
    def __getitem__(self, index):
        sample = dict()

        sample['ppg'] = np.frombuffer(self.lmdbtxn.get("{}-ppg".format(index).encode()), dtype="float32")
        sample['abp'] = np.frombuffer(self.lmdbtxn.get("{}-abp".format(index).encode()), dtype="float32")
        sample['sbp'] = np.frombuffer(self.lmdbtxn.get("{}-sbp".format(index).encode()), dtype="float32")
        sample['dbp'] = np.frombuffer(self.lmdbtxn.get("{}-dbp".format(index).encode()), dtype="float32")
        sample['ecg'] = np.frombuffer(self.lmdbtxn.get("{}-ecg".format(index).encode()), dtype="float32")
        sample['handcrafted'] = np.frombuffer(self.lmdbtxn.get("{}-handcrafted".format(index).encode()), dtype="float32")
        sample['handcrafted-std'] = np.frombuffer(self.lmdbtxn.get("{}-handcrafted-std".format(index).encode()), dtype="float32")
        sample['sample-id'] = index
        sample['case-id'] = self.index_by_sample_id[index][0]
        
        if self.load_spectrogram:
            sample['cwtppg'] = np.reshape(np.frombuffer(self.lmdbtxn.get("{}-cwtppg".format(index).encode()), dtype="float32"), (64, 625))
            sample['cwtecg'] = np.reshape(np.frombuffer(self.lmdbtxn.get("{}-cwtecg".format(index).encode()), dtype="float32"), (64, 625))

        sample['sig'] = np.concatenate((np.expand_dims(sample['ppg'], axis=-1), np.expand_dims(sample['ecg'], axis=-1)), axis=-1)

        # !make sure the ndarray is writeable and owns its own data
        for k in sample:
            sample[k] = np.require(sample[k], requirements=['O', 'W'])
            sample[k].setflags(write=1)

        return sample
    
    def data_columns(self)->list:
        return ['ppg', 'abp', 'sbp', 'dbp', 'ecg', 'cwtppg', 'cwtecg', 'sig', 'handcrafted', 'handcrafted-std']


if __name__ == "__main__":
    #dataset = UCILmdbDataset("./datasource/ucibpds", split_ratio=[0.7, 0.1, 0.2], load_spectrogram=False, mix_cases_in_trainvalid=False)
    #train_sampler, valid_sampler = dataset.get_trainvalidsampler(fold=0)
    #out_list = list()
    #l = 0
    #for case_id in dataset.cases_for_test:
    #    try:
    #        sample_ids = dataset.index_by_case_id[case_id]
    #        if len(sample_ids) > 75:
    #            out_list.append((case_id, sample_ids))
    #            l = l + len(sample_ids)
    #    except:
    #        print("exception on case", case_id)
    #    
    #print(l)
    #print(len(out_list))

    dataset = ICTMatDataset("./datasource/ictbpdsall", subjects=list(range(0, 44)))
    for i in range(dataset.__len__()):
        dataset.__getitem__(i)

    

