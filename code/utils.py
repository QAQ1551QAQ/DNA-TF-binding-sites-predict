
import config
from sklearn.model_selection import train_test_split
import torch
import time
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_curve, precision_score, recall_score

# 实现DNA序列的反向、互补、及反向互补
def DNA_complement(sequence):
    sequence = sequence.upper()
    sequence = sequence.replace('A', 't')
    sequence = sequence.replace('T', 'a')
    sequence = sequence.replace('C', 'g')
    sequence = sequence.replace('G', 'c')
    return sequence.upper()
def DNA_reverse(sequence):
    sequence = sequence.upper()
    return sequence[::-1]
def new_dna(DNA):
    tmp1 = DNA_complement(DNA) #互补序列
    tmp2 = DNA_reverse(DNA) #反向序列
    tmp3 =DNA_reverse(DNA_complement(DNA)) # 反向互补 
    new_dna = DNA + tmp2 + tmp1 + tmp3
    return new_dna

#分词
def word_cut(dna):
    #dna = new_dna(dna) #是否拼接
    cut_list = []
    dna_length = len(dna)
    for i in range(dna_length):
        tmp = dna[i:i + 3]
        if (len(tmp)) == 3:
            cut_list.append(tmp)
    return cut_list

#重新索引
def get_index_drop(list_name):
    for i, name in enumerate(list_name):
        name = name.reset_index(drop=True)
        list_name[i] = name
    return list_name

#根据词典对要加载的预训练词向量按索引排序
def wordIndex_map_w2vIndex(vocab_dic, w2v_dir,pretrain_dir):
    emb_dim = config.embed
    embeddings = np.random.rand(len(vocab_dic), emb_dim)
    fr = open(w2v_dir, "r", encoding='UTF-8')
    for i, line in enumerate(fr.readlines()):
        if i == 0:  # 若第一行是标题，则跳过
            continue
        lin = line.strip().split(" ")
        if lin[0] in vocab_dic:
            idx = vocab_dic[lin[0]]
            emb = [float(x) for x in lin[1:config.embed + 1]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    fr.close()
    np.savez_compressed(pretrain_dir, mapped_index_emmbeddings = embeddings)
    
#建立词表
def build_vocab(train_text, max_size, min_freq):
    vocab_dic = {}
    for word_list in train_text:
        for word in word_list:
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({config.UNK: len(vocab_dic), config.PAD: len(vocab_dic) + 1})
    wordIndex_map_w2vIndex(vocab_dic, config.w2v_dir, config.pretrain_dir) # 词id映射到预训练词向量
    config.embedding_pretrained = torch.tensor(
            np.load('../data/pretrain_embedding_3mer.npz')["mapped_index_emmbeddings"].astype('float32'))
    return vocab_dic

#将数据处理为，格式 [([num,num,...], label, seq_len), ([...], label, seq_len), ...], num为数字，是word对应的字典value
def build_dataset(train_text, train_label, val_text, val_label, test_text, test_label):
    vocab = build_vocab(train_text, max_size=config.max_vocab_size, min_freq=1)
    # print(vocab)
    # print("Vocab size:",len(vocab))
    def load_dataset(text, labels, pad_size):
        contents = []
        for i, token in enumerate(text):
            words_line = [] # save index of words
            seq_len = len(token)
            label = labels[i]
            if pad_size:
                if len(token) < pad_size:
                    token.extend([config.PAD] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            # word to id
            for word in token:
                words_line.append(vocab.get(word, vocab.get(config.UNK)))
                # import pdb;pdb.set_trace()
            contents.append((words_line, int(label), seq_len))
        return contents
    train = load_dataset(train_text, train_label, config.pad_size)
    dev = load_dataset(val_text, val_label, config.pad_size)
    test = load_dataset(test_text, test_label, config.pad_size)
    return vocab, train, dev, test

#iter
class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

#已使用时间
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

#获取iter
def build_iterator(dataset, batch_size, device):
    iter = DatasetIterater(dataset, batch_size, device)
    return iter

def getDataSet(train_index, test_index, text, label):
    train_text = text[train_index]
    train_label = label[train_index]
    test_text = text[test_index]
    test_label = label[test_index]
    #从交叉验证的训练集中划分出验证集
    train_text, val_text, train_label, val_label = train_test_split(
        train_text, train_label, test_size=config.dev_size, stratify=train_label, random_state=config.SEED,shuffle=True)
    #重新进行自然索引
    indexReset = get_index_drop([train_text, train_label, val_text, val_label, test_text, test_label])
    [train_text, train_label, val_text, val_label, test_text, test_label] = indexReset
    #dataSet
    print("dataSets...")
    start_time = time.time()
    vocab, train_data, dev_data, test_data = build_dataset(train_text, train_label, val_text, val_label, test_text, test_label)
    train_iter = build_iterator(train_data, config.batch_size, config.device)
    dev_iter = build_iterator(dev_data, config.batch_size, config.device)
    test_iter = build_iterator(test_data, config.batch_size, config.device)
    time_dif = get_time_dif(start_time)
    print("dataSets already spent time:", time_dif)
    return vocab, train_iter, dev_iter, test_iter 
    # train_iter type is tuple, ((tensor([num, ...]),tensor([seq_len, ...])),tensor(label, ...))


def auc_roc_pr_f1(labels_all, predict_all ,maxprob_all):
     # roc-auc, pr-auc, f1
    fpr, tpr, thresholds = roc_curve(labels_all, maxprob_all, pos_label = 1)
    roc_auc = auc(fpr, tpr)  #auc为Roc曲线下的面积  
    precision, recall, thresholds = precision_recall_curve(labels_all, maxprob_all, pos_label = 1)
    pr_auc = auc(recall, precision)
    f1 = f1_score(labels_all, predict_all, average='weighted')
    pre = precision_score(labels_all, predict_all, average='weighted')
    rec = recall_score(labels_all, predict_all, average='weighted')

    #开始画ROC曲线
    plt.plot(fpr, tpr, 'b',label='ROC-AUC = %0.2f'% roc_auc)
    plt.plot(recall, precision, 'g',label='PR-AUC = %0.2f'% pr_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    # plt.xlabel('False Positive Rate') #横坐标是fpr
    # plt.ylabel('True Positive Rate')  #纵坐标是tpr
    # plt.title('Receiver operating characteristic')
    # plt.show()
    plt.savefig(config.save_picture + 'roc_auc%s.jpg'%roc_auc )
    plt.close()
    return roc_auc, pr_auc, f1, pre, rec

if __name__ == "__main__":
    pass

