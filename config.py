import sys
import os
from os.path import dirname
# custom libs
project_root = dirname(__file__)
sys.path.append(project_root)

datasets_cfg = {
            'ucibp': {
                'lmdb_folder': os.path.join(project_root, "datasource", "ucibpds"),
                'load_spectrogram': False,
                'split_ratio': [0.7, 0.1, 0.2],
                'mix_cases_in_trainvalid': False
            },
        }

models_ft_cfg = {
            'models.PPGECGNet_V0e2x1b':{
                #'path': "./checkpoints/Pretrain-Baselines/Pretrain-Baselines-PPGECGNet_V0e2x1b-2023_04_29-09_01_17/ckpt-best.pth",
                #'path': "./checkpoints/debug/debug-PPGECGNet_V0e2x1b-2023_07_17-14_57_16/ckpt-best.pth",
                'path': "/media/hdd/feiyi/workspace/universal_bloodpressure/checkpoints/Pretrain-Baselines/Pretrain-Baselines-PPGECGNet_V0e2x1b-2023_04_29-09_01_17/ckpt-best.pth",
                'tune': "default"
            }, 
        }
