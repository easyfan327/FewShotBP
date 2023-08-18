'''
Author: Feiyi Fan
Date: 2021-07-02 16:44:56
LastEditTime: 2021-10-26 11:18:45
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /uci_bp/preprocess/generate_lmdb.py
'''
import lmdb
import pickle
import os, sys
import numpy as np
from scipy.io import loadmat

LMDB_MAP_SIZE = 1000 * 1000 * 1000 * 1000  # 1T

dataset_folder = '/mnt/nvmessd/workspace/uci_bp/preprocess/cases_cleaned'
lmdb_folder = '/mnt/nvmessd/workspace/uci_bp/preprocess/lmdb-debug'

lmdbenv = lmdb.open(lmdb_folder, map_size=LMDB_MAP_SIZE)

file_names = [
    f.path for f in os.scandir(dataset_folder)
    if f.is_file() and f.name.endswith(".mat")
]

# sample_id所归属的case_id，以及其在*清理后的*mat文件中的序号
index_by_sample_id = list()
# 每个case_id的mat文件对应的sample_id列表
index_by_case_id = dict()
# 整个lmdb数据集中case_id的列表
case_list = list()

# NOTE:
# ! case_no 不与 case_id对应，因为在清理过程中，某些case_id对应的mat文件被清理掉了

case_n = len(file_names)
case_no = 0

# LMDB数据库中sample的全局id
sample_id = 0

for file_name in file_names:
    mat_file = loadmat(file_name, struct_as_record=False)
    data = mat_file['data'][0][0]
    sample_n_in_current_case = data.data_range.shape[1]


    #case_id = data.caseid[0][0]
    case_id = int(str.split(os.path.basename(file_name), '.')[0])

    case_list.append(case_id)
    index_by_case_id[case_id] = list()

    txn = lmdbenv.begin(write=True)
    for sample_no_in_current_case in range(sample_n_in_current_case):
        print("{:6d}/{:6d}|{:6d} of {:6d}".format(case_no, case_n,
                                                  sample_no_in_current_case,
                                                  sample_n_in_current_case))

        start_idx = data.data_range[0, sample_no_in_current_case]
        end_idx = data.data_range[1, sample_no_in_current_case]
        data_range = range(start_idx, end_idx + 1)

        txn.put(key="{}-ppg".format(sample_id).encode(), value=np.squeeze(data.ppg)[data_range].tobytes())
        txn.put(key="{}-abp".format(sample_id).encode(), value=np.squeeze(data.abp)[data_range].tobytes())
        txn.put(key="{}-sbp".format(sample_id).encode(), value=np.squeeze(data.sbp)[data_range].tobytes())
        txn.put(key="{}-dbp".format(sample_id).encode(), value=np.squeeze(data.dbp)[data_range].tobytes())
        txn.put(key="{}-ecg".format(sample_id).encode(), value=np.squeeze(data.ecg)[data_range].tobytes())
        txn.put(key="{}-handcrafted".format(sample_id).encode(), value=data.handcrafted_features[:, sample_no_in_current_case].tobytes())
        txn.put(key="{}-handcrafted-std".format(sample_id).encode(), value=data.handcrafted_features_std[:, sample_no_in_current_case].tobytes())

        if sample_n_in_current_case != 1:
            txn.put(key="{}-cwtppg".format(sample_id).encode(),
                    value=data.cwtppg[:, :, sample_no_in_current_case].tobytes())
            txn.put(key="{}-cwtecg".format(sample_id).encode(),
                    value=data.cwtecg[:, :, sample_no_in_current_case].tobytes())
        else:
            txn.put(key="{}-cwtppg".format(sample_id).encode(),
                    value=data.cwtppg[:, :].tobytes())
            txn.put(key="{}-cwtecg".format(sample_id).encode(),
                    value=data.cwtecg[:, :].tobytes())

        index_by_sample_id.append((case_id, sample_no_in_current_case))
        index_by_case_id[case_id].append(sample_id)
        sample_id = sample_id + 1

    txn.commit()
    case_no = case_no + 1

# * Writing meta-data / index data

txn = lmdbenv.begin(write=True)
txn.put(key="index_by_sample_id".encode(),
        value=pickle.dumps(index_by_sample_id))
txn.put(key="index_by_case_id".encode(), value=pickle.dumps(index_by_case_id))
txn.put(key="case_list".encode(), value=pickle.dumps(case_list))
txn.commit()
