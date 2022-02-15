
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

def get_data_col(pd_data):
    labels_all= pd_data['labels_all']
    predict_all = pd_data['predict_all']
    prob_all = pd_data['prob_all']
    return labels_all,predict_all,prob_all

def auc_roc_pr_f1(labels_all, predict_all ,maxprob_all):
     # roc-auc, pr-auc, f1
    fpr, tpr, thresholds = roc_curve(labels_all, maxprob_all, pos_label = 1)
    roc_auc = auc(fpr, tpr)  #auc为Roc曲线下的面积  
    precision, recall, thresholds = precision_recall_curve(labels_all, maxprob_all, pos_label = 1)
    pr_auc = auc(recall, precision)
    f1 = f1_score(labels_all, predict_all)
    pre = precision_score(labels_all, predict_all)
    rec = recall_score(labels_all, predict_all)
    return fpr, tpr, recall, precision, roc_auc, pr_auc

def get_values(pd_data):
    # 获取每一列
    labels_all,predict_all,prob_all = get_data_col(pd_data)
    # 计算绘图所需的值
    fpr, tpr, recall, precision, roc_auc, pr_auc = auc_roc_pr_f1(labels_all, predict_all ,prob_all)
    return fpr, tpr, recall, precision, roc_auc, pr_auc

def draw_roc_pr(fpr_1, tpr_1, recall_1, precision_1, roc_auc_1, pr_auc_1,
                fpr_2, tpr_2, recall_2, precision_2, roc_auc_2, pr_auc_2,
                fpr_3, tpr_3, recall_3, precision_3, roc_auc_3, pr_auc_3,
                fpr_4, tpr_4, recall_4, precision_4, roc_auc_4, pr_auc_4):
    # ROC AUC
    pl = plt.figure(figsize=(14,6), )
    ax1 = pl.add_subplot(1,2,1)
    plt.title('ROC AUC')
    plt.plot(fpr_2, tpr_2, 'm',label='DeepBind = %0.3f'% roc_auc_2)
    plt.plot(fpr_3, tpr_3, 'g',label='DanQ = %0.3f'% roc_auc_3)
    plt.plot(fpr_4, tpr_4, 'c',label='DeepD2V = %0.3f'% roc_auc_4)
    plt.plot(fpr_1, tpr_1, 'b',label='ResHybridDeep = %0.3f'% roc_auc_1)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    # plt.show()
    # PR AUC
    ax2 = pl.add_subplot(1,2,2)
    plt.title('PR AUC')
    plt.plot(recall_2, precision_2, 'm',label='DeepBind = %0.3f'% pr_auc_2)
    plt.plot(recall_3, precision_3, 'g',label='DanQ = %0.3f'% pr_auc_3)
    plt.plot(recall_4, precision_4, 'c',label='DeepD2V = %0.3f'% pr_auc_4)
    plt.plot(recall_1, precision_1, 'b',label='ResHybridDeep = %0.3f'% pr_auc_1)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0,1])
    plt.ylim([0,1])

    plt.show()


if __name__ == "__main__":
    # pd_data_1 = np.load('../res/cnnLstmAtt_11_12-22_01.08_0.9463671910233717.npz') # ResHybridDeep 
    # print(pd_data_1.files) # ['labels_all', 'predict_all', 'prob_all']  #Gm12878Batf
    fpr_1, tpr_1, recall_1, precision_1, roc_auc_1, pr_auc_1 = get_values(np.load('../res/cnnLstmAtt_11_12-22_01.08_0.9463671910233717.npz'))# ResHybridDeep
    fpr_2, tpr_2, recall_2, precision_2, roc_auc_2, pr_auc_2 = get_values(np.load('../res/DeepBind_12-21_23.04_0.8950779870723408.npz'))# DeepBind
    fpr_3, tpr_3, recall_3, precision_3, roc_auc_3, pr_auc_3 = get_values(np.load('../res/DanQ_12-21_21.10_0.8953318062815945.npz'))# DanQ
    fpr_4, tpr_4, recall_4, precision_4, roc_auc_4, pr_auc_4 = get_values(np.load('../res/DeepD2V_10kfold_12-21_18.23_0.9302605295043288.npz'))# DeepD2V
    
    # draw
    draw_roc_pr(fpr_1, tpr_1, recall_1, precision_1, roc_auc_1, pr_auc_1,
                fpr_2, tpr_2, recall_2, precision_2, roc_auc_2, pr_auc_2,
                fpr_3, tpr_3, recall_3, precision_3, roc_auc_3, pr_auc_3,
                fpr_4, tpr_4, recall_4, precision_4, roc_auc_4, pr_auc_4
                 )
