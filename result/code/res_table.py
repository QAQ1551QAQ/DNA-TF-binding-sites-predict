import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

regex  = re.compile(r"^(Gm12878|K562|H1hesc)(.*?)(ak20|101388|sc(.*?)|Pcr(.*?)|V041(.*?))")
def get_str(str):
    res = regex.findall(str)
    return res

def get_res(path):
    data_all = pd.read_csv(path, header=None, encoding='utf-8')
    # print(data_all.head(2)) # ['name', '1','roc_auc', '3', 'pr_auc', '5', 'pre', ''7, 'rec', '9', 'acc', '11', 'f1']
    data_all.columns=['name', '1','roc_auc', '3', 'pr_auc', '5', 'pre', '7', 'rec', '9', 'acc', '11', 'f1']
    # print(data_all.head(2))
    data_res = data_all.drop(columns = ['1', '3', '5', '7', '9', '11'])
    # print(data_res.head(2))
    data_res['Cell Line'] = data_res['name'].apply(lambda x: get_str(x)[0][0])
    data_res['TF'] = data_res['name'].apply(lambda x: get_str(x)[0][1])
    data_res = data_res.drop(columns = ['name'])
    order = ['Cell Line', 'TF', 'roc_auc', 'pr_auc', 'pre', 'rec', 'acc','f1']
    data_res = data_res[order]
    data_res.sort_values(['Cell Line', 'TF'], inplace = True)
    return data_res

def get_roc_pr_f1(roc_pr_f1_name):
    DeepBind_roc= DeepBind_res[['Cell Line','TF', roc_pr_f1_name]].reset_index(drop=True)  #'roc_auc'
    DanQ_roc = DanQ_res[[roc_pr_f1_name]].reset_index(drop=True) 
    DeepD2V_roc = DeepD2V_res[[roc_pr_f1_name]].reset_index(drop=True)
    DeepSEA_roc = DeepSEA_res[[roc_pr_f1_name]].reset_index(drop=True)
    att_roc = att_res[[roc_pr_f1_name]].reset_index(drop=True) 
    result =  pd.concat([DeepBind_roc, DanQ_roc, DeepD2V_roc, DeepSEA_roc, att_roc], axis=1)
    result.columns =['Cell Line','TF','DeepBind', 'DanQ', 'DeepD2V', 'DeepSEA', 'CRA-KAN']
    print(result)
    result.to_csv('../figureAndResult/res_table_%s.csv'%roc_pr_f1_name, index=None)

if __name__ == "__main__":
    path1 = '../res/DeepBind_res.txt'
    path2 = '../res/DanQ_res.txt'
    path3 = '../res/DeepD2V_res.txt'
    path4 = '../res/DeepSEA_res.txt'
    path5 = '../res/CRA-KAN_res.txt' # CRA-KAN
    
    DeepBind_res = get_res(path1)
    DanQ_res = get_res(path2)
    DeepD2V_res = get_res(path3)
    DeepSEA_res = get_res(path4)
    att_res = get_res(path5)

    get_roc_pr_f1('roc_auc')
    get_roc_pr_f1('pr_auc')
    get_roc_pr_f1('f1')
    get_roc_pr_f1('acc')

    pass