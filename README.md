# FewShotBP
The repository contains open-source code for IMWUT 2023 paper “FewShotBP: Towards Personalized Ubiquitous Continuous Blood Pressure Measurement”

# Environment Configuration
The environment can be prepared using conda package managing.
```
conda env create -f environment.yml
```
Please refer to [Managing environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-from-file) for detailed information

# Run Experiments

## Prepare Dataset

1. **Option 1**: The cleaned dataset can be downloaded from [Hugging Face](https://huggingface.co/datasets/Feiyi/FewShotBP) and the mdb file should be placed in ```./datasource/ucibpds/```
2. **Option 2**: Please refer to section below to build dataset from scratch. 

## Run Experiments with Network Finetuning

```
python transfer_finetune.py -n N -e EXP_NAME
```
, where ```N``` is the number of samples for personalization. The experiments are logged in ```./tensorboard/EXP_NAME/RUN_NAME```, and can be visualized using tensorboard. Please refer to ```HPARAMS``` section in tensorboard to see hyperparameters and obtained results. 

In addition, the results (ground truth and predictions for every subjects) will be dumped to ```./tensorboard/EXP_NAME/RUN_NAME/perf_train_stat_group.trialdump``` and ```./tensorboard/EXP_NAME/RUN_NAME/perf_valid_stat_group.trialdump```

## Run Experiments with Proposed Personalization Adapter

```
python transfer_pa.py -e EXP_NAME
```
The scriptss will run personalization with 5, 10, 25, 50 samples sequentially, and the results (ground truth and predictions) will be dumped to ```./records/pa2ucibp/```

To print the statistics, run ```show_pa_results.ipynb```. Note, the argument of ```read_pa_trial_dump()``` should be changed to file name of generated dump file in previous step.

# Build Dataset from Scratch

1. Place the UCIBP dataset (derived from MIMIC-II) in ```./datasource/ucibp/```, the dataset can be downloaded from [Cuff-Less Blood Pressure Estimation](https://archive.ics.uci.edu/dataset/340/cuff+less+blood+pressure+estimation).
2. Split the dataset into independent case files using MATLAB script ```generate_cases.m```.
3. Clean the independent case files using MATLAB script ```clean_dataset.m```.
4. Generate LMDB dataset by
```
python generate_lmdb.py
```

Note, the ```dataset_folder``` and ```lmdb_folder``` variable in ```generate_lmdb.py``` should be changed to actual paths of cleaned independent case files and desired output lmdb data folder.

# Pre-train Models on UCIBP dataset

The model structure of MSTNN is defined in ```./models/PPGECGNet_V0e2x1b.py```. Run the following code to pre-train the model.
```
python train.py
```

The model file can be replaced with your own implementations.
