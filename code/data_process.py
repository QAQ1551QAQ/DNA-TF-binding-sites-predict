#coding=utf-8
import os
import pandas as pd

# 解压的 50 ChIP-seq datasets 数据存放目录
dataFiles = '../data/rawdata/'

#接收标签[0,1]、文件名['negative.fasta','positive.fasta']、文件夹地址，返回提取好的数据
def tmp_get_data(label_0_1, neg_or_pos, tmp_root_dir):
    tmp_data = pd.read_csv(tmp_root_dir +neg_or_pos, names=['text']) # negative label is 0  # positive label is 1
    tmp_data["text_len"] = tmp_data['text'].str.len()
    tmp_data['label']= label_0_1
    tmp_data = tmp_data[tmp_data['text_len']==200]
    return tmp_data[['label','text']]

#查询文件及存储
def data_get(filePath):
    for root, dirs, files in os.walk(filePath):
        for dir in dirs:
            tmp_root_dir = os.path.join(root,dir)
            tmp_list = ['/negative.fasta','/positive.fasta']
            data = pd.DataFrame()
            for  label_0_1, neg_or_pos in enumerate(tmp_list):
                temp_df = tmp_get_data(label_0_1, neg_or_pos, tmp_root_dir)
                data = data.append(temp_df,ignore_index=False,verify_integrity=False) #ignore_index 如果为 True 则重新进行自然索引
            data.to_csv('../data/processed/' + dir[16:] +'.csv',index=None)
            print(str(dir[16:]) + '.csv 已经处理完成')
            
    return -1


if __name__ == '__main__':
    # data_get(dataFiles) # 提取 50 ChIP-seq datasets 数据。

    pass