#!/usr/bin/env bash
rm -rf ./logs_run/*
rm -rf ../log/*
# rm -rf ../test_report_logs/*
# rm -rf ../model_saved_dict/*
# rm -rf ../save_fig/*
# rm -rf ../data/res/*
# rm -rf ../data/savePictureData/*
# rm -rf ../data/save_roc_plot/*

#run.py
#nohup python3 run.py >> logs_run/textCNN.log 2>&1 &
#nohup python3 run.py >> logs_run/kan.log 2>&1 &
nohup python3 run_all.py >> logs_run/run_all.log 2>&1 &
#nohup python3 run.py >> logs_run/cnnLstmAtt_11.log 2>&1 &
#nohup python3 run.py >> logs_run/cnnLstmAtt.log 2>&1 &
#nohup python3 run.py >> logs_run/textLstmAtt.log 2>&1 &
#nohup python3 run.py >> logs_run/textLstmCnn.log 2>&1 &
#nohup python3 run.py >> logs_run/transformer.log 2>&1 &
#nohup python3 run.py >> logs_run/dpCnn.log 2>&1 &
#nohup python3 run.py >> logs_run/DanQ.log 2>&1 &
#nohup python3 run.py >> logs_run/DeepBind.log 2>&1 &
