# DNA-TF-binding-sites-predict
transcription factor binding sites; residual learning; attention mechanism; CNN; BiLSTM; KAN

# Environmental requirements：
  1. Python 3.9.7
  2. pytorch 2.2.2+cu121
  3. See requirements.txt for other python packages that need to be installed.

# required data：
  1. rawdata download link： 链接:https://pan.baidu.com/s/1xPzWjk3xNQJzEHcpysIjgw  密码:usch
  2. Word vector 3mer.txt download link： 链接:https://pan.baidu.com/s/1EN5bxXcKACGzpKPmQJ-QHg  密码:cie9

# Operation mode:
  1. Download 50 public ChIP-seq data and save it to data/rawdata, run code/data_process.py to process the data into the corresponding format.
  2. Modify the corresponding hyperparameter code/config.py according to actual needs.
  3. Run code/run_all.py to train the model.

# Results display：
  1. result/res saves the running result of each model
  2. The code for each performance metric calculation is provided in results/code

# Code file description
├── config.py # Modify hyperparameters

├── data_process.py # Data preprocessing

├── logs_run # log file

├── models # models

│ └── DeepBind.py              # Single-layer CNN + Two-layer MLP
│ └── DeepBind-KAN.py          # Single-layer CNN + KAN
│ └── DanQ.py                  # Single-layer CNN + BiLSTM + Two-layer MLP
│ └── DanQ-KAN.py              # Single-layer CNN + BiLSTM + KAN
│ └── DeepD2V.py               # Three-layer CNN + BiLSTM + Two-layer MLP
│ └── DeepD2V-KAN.py           # Three-layer CNN + BiLSTM + KAN
│ └── DeepSEA.py               # Three-layer CNN + Two-layer MLP
│ └── DeepSEA-KAN.py           # Three-layer CNN + KAN
│ └── RA-KAN.py                # BiLSTM + attention + KAN
│ └── CRA-KAN(ConvBlock1).py   # ConvBlock1 + BiLSTM + attention + KAN
│ └── CRA-KAN.py               # ConvBlock1/2/3 concatenation +  BiLSTM + attention + KAN

├── run_all.py # run all datasets

├── start.sh # run multiple commands at once

├── train_eval.py # training, validation

└── utils.py # Common functions
