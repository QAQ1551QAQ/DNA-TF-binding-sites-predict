# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter
import utils   

def init_network(model, method='xavier', exclude='embedding', seed=2021):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    lambda1 = lambda epoch: 0.95 ** epoch # 第二组参数的调整方法
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1) # 选定调整方法
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size =10, gamma=0.1)
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    #dev_best_loss = float('inf')
    dev_best_roc = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss, roc_auc = evaluate(config, model, dev_iter)
                #if dev_loss < dev_best_loss:
                #    dev_best_loss = dev_loss
                if dev_best_roc < roc_auc:
                    dev_best_roc = roc_auc
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val roc_auc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, roc_auc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
        scheduler.step() #学习率衰减
    writer.close()
    roc_auc, pr_auc, pre, rec, test_acc,f1 = test(config, model, test_iter)
    return roc_auc, pr_auc, pre, rec, test_acc,f1

def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion, roc_auc,pr_auc,f1,pre,rec= evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    print('roc_auc: {},  pr_auc: {},  f1: {}'.format(roc_auc, pr_auc, f1))
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # test结果写入文件
    with open(config.test_report_logs,"a",encoding="utf-8") as fw:
        fw.write(msg.format(test_loss, test_acc))
        fw.write(test_report)
        fw.write('roc_auc: {},  pr_auc: {},  f1: {}'.format(roc_auc, pr_auc, f1))
    return roc_auc, pr_auc, pre, rec, test_acc,f1


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    prob_all = np.array([], dtype=np.float64)
    # prob_all_tmp = [];lable_all_tmp = []; predict_all_tmp=[]# tmp
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts) 
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy() #行的最大值的index
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            # softmax
            outputs_prob = F.softmax(outputs, 1)   # sotfmax 计算概率值
            prob = outputs_prob.data.cpu().numpy() # 正例概率值
            prob_all = np.append(prob_all, prob[:,1])    # 存值 
            #tmp
            # prob_all_tmp.extend(maxprob);lable_all_tmp.extend(labels); predict_all_tmp.extend(predic)
            
    acc = metrics.accuracy_score(labels_all, predict_all)
    roc_auc, pr_auc, f1, pre, rec = utils.auc_roc_pr_f1(labels_all, predict_all ,prob_all)
# roc = metrics.roc_auc_score(labels_all, prob_all, multi_class='ovr',average="weighted")
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        # np.savez_compressed('./tmp_lable_prob',label=labels_all,prob=maxprob_all,pre=predict_all, sftmxpre = idx_sftmx_all)
        # roc-auc, pr-auc, f1
        roc_auc, pr_auc, f1, pre, rec = utils.auc_roc_pr_f1(labels_all, predict_all ,prob_all)
        return acc, loss_total / len(data_iter), report, confusion, roc_auc, pr_auc, f1, pre, rec
    return acc, loss_total / len(data_iter), roc_auc
