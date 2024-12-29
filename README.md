# DNA-TF-binding-sites-predict
transcription factor binding sites; CNN; BiLSTM; KAN; residual connections

# Environmental requirements：
  1. Python 3.12.7
  2. pytorch 2.5.0
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

├── efficient_kan # kan code

├── models # models

│ └── C-KAN.py                 # (dna2vec) + C(ConvBlock1/2/3) + KAN

│ └── C-KAN-nonD2v.py          # (Non-dna2vec) + C(ConvBlock1/2/3) + KAN

│ └── C-KAN-ConvBlock1.py      # (dna2vec) + C(ConvBlock1) + KAN

│ └── C-KAN-ConvBlock2.py      # (dna2vec) + C(ConvBlock2) + KAN

│ └── C-KAN-ConvBlock3.py      # (dna2vec) + C(ConvBlock3) + KAN

│ └── C-KAN-ConvBlock2-k2.py   # (dna2vec) + C(ConvBlock2)-kernel*2 + KAN

│ └── C-KAN-ConvBlock2-k3.py   # (dna2vec) + C(ConvBlock2)-kernel*3 + KAN

│ └── CBR-KAN.py               # (dna2vec) + C(ConvBlock1/2/3) + BiLSTM + Residual + KAN

│ └── CB-KAN_ no-Residual.py   # (dna2vec) + C(ConvBlock1/2/3) + BiLSTM + KAN

│ └── CBR-MLP.py               # (dna2vec) + C(ConvBlock1/2/3) + BiLSTM + Residual + MLP

│ └── DeepBind.py              # Single-layer CNN + Two-layer MLP

│ └── DanQ.py                  # Single-layer CNN + BiLSTM + Two-layer MLP

│ └── DeepD2V.py               # Three-layer CNN + BiLSTM + Two-layer MLP

│ └── DeepSEA.py               # Three-layer CNN + Two-layer MLP

│ └── CNN.py                   # CNN

│ └── BiLSTM.py                # BiLSTM

│ └── BiGRU.py                 # BiGRU

├── run_all.py # run all datasets

├── start.sh # run multiple commands at once

├── train_eval.py # training, validation

└── utils.py # Common functions
