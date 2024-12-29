import config
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import utils
from train_eval import train,init_network
from importlib import import_module
import numpy as np
import torch
import os, time

np.random.seed(config.SEED)
torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

if __name__ == '__main__':
    # file = 'Gm12878BatfPcr1xPkRep1'
    start_time = time.time()
    #filePath = '../data/tmp-processed/'
    filePath = '../data/processed/'
    for root, dirs, files in os.walk(filePath): 
        for file in files:
            one_of_file = os.path.join(root, file)
            print(one_of_file)

            one_of_data = pd.read_csv(one_of_file, encoding='utf-8')
            one_of_data['text_cut'] = one_of_data['text'].apply(lambda x: utils.word_cut(x))
            label = one_of_data['label'] # <class 'pandas.core.series.Series'>
            text = one_of_data['text_cut'] # <class 'pandas.core.series.Series'>
            #交叉验证
            kf = StratifiedKFold(n_splits=config.k_fold, shuffle=True, random_state=config.SEED)
            k = 0; mcc_all = 0; roc_auc_all=0; pr_auc_all_all=0; pre_all=0; rec_all=0; test_acc_all=0;f1_all=0

            for train_index, test_index in kf.split(text, label):
                vocab, train_iter, dev_iter, test_iter = utils.getDataSet(train_index, test_index, text, label)
                # train
                config.len_vocab = len(vocab) #词表大小
                someModel = import_module('models.' + config.model_name)
                model = someModel.Model(config).to(config.device)
                #print(model);print(vocab)
                #init_network(model)
                mcc,roc_auc, pr_auc, pre, rec, test_acc, f1 = train(config, model, train_iter, dev_iter, test_iter, file)

                k += 1; mcc_all += mcc;roc_auc_all += roc_auc; pr_auc_all_all += pr_auc; pre_all+=pre; rec_all+=rec; test_acc_all+=test_acc;f1_all+=f1
                if (k==1):
                    break
                # test结果写入文件
            ave_mcc = mcc_all/k; ave_roc_auc = roc_auc_all/k; ave_pr_auc=pr_auc_all_all/k; ave_pre=pre_all/k; ave_rec=rec_all/k; ave_test_acc=test_acc_all/k; ave_f1=f1_all/k
            with open(config.save_res, "a", encoding="utf-8") as fw:
                fw.write(('%s,roc_auc: ,{},  pr_auc: ,{}, pre: ,{}, rec: ,{}, acc: ,{}, f1: ,{}, mcc: ,{}\n'%file).format(ave_roc_auc, ave_pr_auc, ave_pre, ave_rec, ave_test_acc, ave_f1, ave_mcc))
    time_dif = utils.get_time_dif(start_time)
    print("run all spent time:", time_dif)
