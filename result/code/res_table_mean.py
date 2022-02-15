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


if __name__ == "__main__":
    path1 = '../res/DeepBind_res.txt'
    path2 = '../res/DanQ_res.txt'
    path3 = '../res/DeepD2V_10kfold_L_res.txt'
    path4 = '../res/cnnLstmAtt_11_res.txt' # ResHybridDeep
    
    DeepBind_res = get_res(path1)
    DanQ_res = get_res(path2)
    DeepD2V_res = get_res(path3)
    att_res = get_res(path4)

    DeepBind_mean = pd.DataFrame(DeepBind_res.mean()).T
    DanQ_mean = pd.DataFrame(DanQ_res.mean()).T
    DeepD2V_mean = pd.DataFrame(DeepD2V_res.mean()).T
    att_mean = pd.DataFrame(att_res.mean()).T

    result =  pd.concat([DeepBind_mean, DanQ_mean, DeepD2V_mean, att_mean], axis=0)
    result.index = ['DeepBind','DanQ', 'DeepD2V', 'ResHybridDeep']
    print(result)
    # result.to_csv('../res/res_table_mean.csv')
    res_plt = result[["f1", "acc", "pr_auc", "roc_auc"]]
    tick_label=['DeepBind','DanQ', 'DeepD2V', 'ResHybridDeep']
    width = 0.42
    x = np.arange(0,8,2)

    fig = plt.figure(figsize = (12,8))
    plt.bar(x, res_plt.loc[:,'f1'],width, label='F1-score',color="c",alpha=0.5)
    for i, j in zip(x, res_plt.loc[:,'f1']):
        plt.text(i+0.001,j+0.001, '%.3f'%j, ha='center')

    plt.bar(x+ width, res_plt.loc[:,'acc'], width, label='Accuracy',color="b", alpha=0.5)
    for i, j in zip(x+ width, res_plt.loc[:,'acc']):
        plt.text(i+0.001,j+0.001, '%.3f'%j, ha='center')

    plt.bar(x+ 2*width, res_plt.loc[:,'pr_auc'], width,label='PR AUC', color="y",alpha=0.5)
    for i, j in zip(x+ 2*width, res_plt.loc[:,'pr_auc']):
        plt.text(i+0.001,j+0.001, '%.3f'%j, ha='center')

    plt.bar(x+ 3*width, res_plt.loc[:,'roc_auc'], width,label='ROC AUC',color="chocolate",alpha=0.7)
    for i, j in zip(x+ 3*width, res_plt.loc[:,'roc_auc']):
        plt.text(i+0.001,j+0.001, '%.3f'%j, ha='center')

    plt.ylim(0.8,0.96)
    plt.legend()
    plt.xticks(x+1.5*width,tick_label)
    plt.show()

    # DeepBind_roc= DeepBind_res[['Cell Line','TF','roc_auc']]
    # DanQ_roc = DanQ_res[['roc_auc']]
    # DeepD2V_roc = DeepD2V_res[['roc_auc']]

    # result =  pd.concat([DeepBind_roc, DanQ_roc, DeepD2V_roc], axis=1)
    # result.columns = ['Cell Line','TF','DeepBind', 'DanQ', 'DeepD2V']
    # print(result)
    pass