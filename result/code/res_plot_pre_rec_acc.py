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
    data_all.columns=['name', '1','roc_auc', '3', 'pr_auc', '5', 'pre', '7', 'rec', '9', 'acc', '11', 'f1', '12', 'mcc']
    # print(data_all.head(2))
    data_res = data_all.drop(columns = ['1', '3', '5', '7', '9', '11', '12'])
    # print(data_res.head(2))
    data_res['Cell Line'] = data_res['name'].apply(lambda x: get_str(x)[0][0])
    data_res['TF'] = data_res['name'].apply(lambda x: get_str(x)[0][1])
    data_res = data_res.drop(columns = ['name'])
    order = ['Cell Line', 'TF', 'roc_auc', 'pr_auc', 'pre', 'rec', 'acc','f1', 'mcc']
    data_res = data_res[order]
    data_res.sort_values(['Cell Line', 'TF'], inplace = True)
    return data_res


if __name__ == "__main__":
    path0 = '../res/exist50/CR-KAN_res.txt' # CR-KAN_res
    path1 = '../res/exist50/DeepBind_res.txt'   # DeepBind_res
    path2 = '../res/exist50/DanQ_res.txt'   # DanQ_res
    path3 = '../res/exist50/DeepD2V_res.txt' # DeepD2V_res
    path4 = '../res/exist50/DeepSEA_res.txt' # DeepSEA_res
    att_res = get_res(path0) # att
    DeepBind_res = get_res(path1)
    DanQ_res = get_res(path2)
    DeepD2V_res = get_res(path3)
    DeepSEA_res = get_res(path4)
    print(att_res)
    # x = np.linspace(0, 1, 50)
    # y = cnnLstmAtt_1_res['roc_auc']
    # x = cnnLstm_res['roc_auc']
    # fig = plt.figure()
    # plt.plot([0.82,1], [0.82,1], ls="--", c=".3")
    # plt.scatter(x, y, s=15)
    # plt.show()
    
    # PR
    pl = plt.figure(figsize=(12,8), dpi=100)
    ax1 = pl.add_subplot(3,4,1)
    plt.title('Precision')
    plt.xlabel('DeepBind')
    plt.ylabel('CBR-KAN')
    plt.plot([0.5,1], [0.5,1], ls="--", c=".3")
    y = att_res['pre']
    x = DeepBind_res['pre']
    plt.scatter(x, y, s=10)
    plt.tight_layout()
    plt.subplots_adjust()
    # plt.show()

    # Rec
    ax4 = pl.add_subplot(3,4,5)
    plt.title('Recall')
    plt.xlabel('DeepBind')
    plt.ylabel('CBR-KAN')
    plt.plot([0.5,1], [0.5,1], ls="--", c=".3")
    y = att_res['rec']
    x = DeepBind_res['rec']
    plt.scatter(x, y, s=10)
    plt.tight_layout()

    # plt.show()

    # acc
    ax7 = pl.add_subplot(3,4,9)
    plt.title('Accuracy')
    plt.xlabel('DeepBind')
    plt.ylabel('CBR-KAN')
    plt.plot([0.7,1], [0.7,1], ls="--", c=".3")
    y = att_res['acc']
    x = DeepBind_res['acc']
    plt.scatter(x, y, s=10)
    plt.tight_layout()
    # plt.show()


    ax2 = pl.add_subplot(3,4,2)
    plt.title('Precision')
    plt.xlabel('DanQ')
    plt.ylabel('CBR-KAN')
    plt.plot([0.5,1], [0.5,1], ls="--", c=".3")
    y = att_res['pre']
    x = DanQ_res['pre']
    plt.scatter(x, y, s=10)
    plt.tight_layout()
    # plt.show()

    # PR AUC
    ax5 = pl.add_subplot(3,4,6)
    plt.title('Recall')
    plt.xlabel('DanQ')
    plt.ylabel('CBR-KAN')
    plt.plot([0.5,1], [0.5,1], ls="--", c=".3")
    y = att_res['rec']
    x = DanQ_res['rec']
    plt.scatter(x, y, s=10)
    plt.tight_layout()

    # plt.show()

    # f1
    ax8 = pl.add_subplot(3,4,10)
    plt.title('Accuracy')
    plt.xlabel('DanQ')
    plt.ylabel('CBR-KAN')
    plt.plot([0.7,1], [0.7,1], ls="--", c=".3")
    y = att_res['acc']
    x = DanQ_res['acc']
    plt.scatter(x, y, s=10)
    plt.tight_layout()

    # ROC AUC
    ax3 = pl.add_subplot(3,4,3)
    plt.title('Precision')
    plt.xlabel('DeepD2V')
    plt.ylabel('CBR-KAN')
    plt.plot([0.5,1], [0.5,1], ls="--", c=".3")
    # y = att_res['pre']
    # x = DeepD2V_res['pre']
    x = DeepD2V_res['pre'][DeepD2V_res['pre'] > 0.75]
    y = att_res['pre'][x.index]
    plt.scatter(x, y, s=10)
    plt.tight_layout()
    # plt.show()

    # PR AUC
    ax6 = pl.add_subplot(3,4,7)
    plt.title('Recall')
    plt.xlabel('DeepD2V')
    plt.ylabel('CBR-KAN')
    plt.plot([0.5,1], [0.5,1], ls="--", c=".3")
    # y = att_res['rec']
    x = DeepD2V_res['rec'][DeepD2V_res['rec'] > 0.75]
    y = att_res['rec'][x.index]
    # x = DeepD2V_res['rec']
    plt.scatter(x, y, s=10)
    plt.tight_layout()

    # plt.show()

    # f1
    ax9 = pl.add_subplot(3,4,11)
    plt.title('Accuracy')
    plt.xlabel('DeepD2V')
    plt.ylabel('CBR-KAN')
    plt.plot([0.7,1], [0.7,1], ls="--", c=".3")
    # y = att_res['acc']
    # x = DeepD2V_res['acc']
    x = DeepD2V_res['acc'][DeepD2V_res['acc'] > 0.75]
    y = att_res['acc'][x.index]
    plt.scatter(x, y, s=10)
    plt.tight_layout()

    # ROC AUC
    ax3 = pl.add_subplot(3,4,4)
    plt.title('Precision')
    plt.xlabel('DeepSEA')
    plt.ylabel('CBR-KAN')
    plt.plot([0.5,1], [0.5,1], ls="--", c=".3")
    y = att_res['pre']
    x = DeepSEA_res['pre']
    plt.scatter(x, y, s=10)
    plt.tight_layout()
    # plt.show()

    # PR AUC
    ax6 = pl.add_subplot(3,4,8)
    plt.title('Recall')
    plt.xlabel('DeepSEA')
    plt.ylabel('CBR-KAN')
    plt.plot([0.5,1], [0.5,1], ls="--", c=".3")
    y = att_res['rec']
    x = DeepSEA_res['rec']
    plt.scatter(x, y, s=10)
    plt.tight_layout()

    # plt.show()

    # f1
    ax9 = pl.add_subplot(3,4,12)
    plt.title('Accuracy')
    plt.xlabel('DeepSEA')
    plt.ylabel('CBR-KAN')
    plt.plot([0.7,1], [0.7,1], ls="--", c=".3")
    y = att_res['acc']
    x = DeepSEA_res['acc']
    plt.scatter(x, y, s=10)
    plt.tight_layout()

    plt.show()
    pass