#conding=utf-8
import torch
import numpy as np
#模型
model_name ='CBR-KAN'
save_path = '../model_saved_dict/' + model_name + '.ckpt'        # 模型训练结果
log_path = '../log/' + model_name
w2v_dir = '../data/3mer.txt'
pretrain_dir = '../data/pretrain_embedding_3mer'
test_report_logs = '../test_report_logs/' + model_name +'_test_report.txt'
save_picture = '../save_fig/'  + model_name
save_res = '../data/res/' + model_name + '_res.txt'
# embedding_pretrained = None
# embedding_pretrained = torch.tensor(
#             np.load('../data/pretrain_embedding_3mer.npz')["mapped_index_emmbeddings"].astype('float32'))
embedding_pretrained = -1               # 运行时赋值
require_improvement = 5000              ## 若超过1000batch效果还没提升，则提前结束训练
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cuda:0'
device = 'cuda:0'
k_fold = 10     # 交叉验证
test_size = 0.1 # 划分出的验证集大小，用来调整模型参数
SEED = 2021    # 随机种子
BATCH_SIZE = 64 # batch_size
batch_size = 64
PAD_SIZE = 198 # 句子长度
pad_size = 198
MAX_VOCAB_SIZE = 10000  # 词表长度限制
max_vocab_size = 10000 
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号

# CNN
num_epochs = 20
len_vocab = -1       # 词表大小, 运行时赋值
embed = 100          # Embedding_size
num_filters = 256     # 卷积核数量
filter_sizes = [1, 3, 5, 7]    # 卷积核大小
dropout = 0.2
num_classes = 2      # 类别数量
learning_rate = 0.0005
class_list = ['negative 0', 'positive 1']         # 类别名称

# RNN
hidden_size = 64
hidden_size2 = 64
num_layers = 2


dim_model = 100 
hidden = 256
num_head = 5
num_encoder = 3




head=4
n_layer=2
emd_dim=100 
d_model=32 
d_ff=256
output_dim=2
dropout=0.2


#fastText
n_gram_vocab = 66
