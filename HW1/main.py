#数值运算
import math
import numpy as np

#读/写数据
import pandas as pd
import os
import csv

#训练过程可视化
from tqdm import tqdm

#Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

#可视化学习曲线
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("wtt")  # 将条目直接写入 log_dir 中的事件文件以供 TensorBoard 使用，tensorboard是可视化工具

def same_seed(seed):
    '''固定随机种子，保证模型初始化参数相同'''
    #保证使用确定性的卷积算法
    torch.backends.cudnn.deterministic = True
    #保证不使用选择卷积算法的机制
    torch.backends.cudnn.benchmark = False
    #为random模块固定随机序列
    np.random.seed(seed)
    #为cpu固定随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        #为所有的GPU设置随机数种子
        torch.cuda.manual_seed_all(seed)

def train_valid_split(data_set, valid_ratio, seed):
    '''将训练数据集分为训练集和验证集'''
    #验证集size 占比*总size
    valid_set_size = int(valid_ratio * len(data_set))
    #训练集的size
    train_set_size = len(data_set) - valid_set_size
    #random_split（总长，[size1，size2]，generator）不固定生成器则每次重新生成的数据集内容不相同，数据集是Subset格式
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    #转换成numpy数组的格式
    return np.array(train_set), np.array(valid_set)


class COVID19Dataset(Dataset):
    '''
    x: 输入的特征.
    y: 结果, 如果没有则做预测.
    '''
    def __init__(self, x, y=None):#返回对象dataset（self.x,self.y）
        if y is None:
            self.y = y #y=none
        else:
            self.y = torch.FloatTensor(y)#转tensor格式
        self.x = torch.FloatTensor(x)#转tensor格式

    def __getitem__(self, idx): #获取某行数据
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):# 获取输入数据的长度
        return len(self.x)


def select_feat(train_data, valid_data, test_data, select_all=True):
    '''选择有用的特征进行回归预测'''
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]  # 最后一列的所有行
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data  # 除去最后一列的所有行

    if select_all:
        feat_idx = list(range(1,88))  # 共多少行
    else:
        feat_idx = list(range(1,35))+[52, 70]  # TODO: 选择合适的特征：例如[range(38,raw_x_train.shape[1]-1)]去掉id和洲

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid


device = 'cuda' if torch.cuda.is_available() else 'cpu'#选择device优先使用GPU
config = {                #这是一个包含训练相关超参数的配置字典。
    'seed': 5201314,      # 可以选择任意一个你喜欢的数字，来固定随机序列
    'select_all': True,   # 确定是否使用所有的特征（可能不是所有的特征都对预测有用，若不全部使用则需进行进一步处理）
    'valid_ratio': 0.2,   # 验证集大小 = 训练集大小 * valid_ratio
    'n_epochs': 3000,     # 总共跑多少轮.
    'batch_size': 256,    #一个batch里含多少数据条目
    'learning_rate': 1e-5, #学习率
    'early_stop': 400,    # 如果有400轮里模型精度不再提升则提前停止
    'save_path': './models/model.ckpt'  # 保存模型路径文件
}

# 设置随机种子固定随机序列
same_seed(config['seed'])

# train_data size训练集大小: 2699 x 118 (id + 37 洲（one-hot） + 16 features特征 x 5天)
# test_data size: 1078 x 117 (没有最后一天的阳性率)
train_data, test_data = pd.read_csv('./covid_train.csv').values, pd.read_csv('./covid_test.csv').values#读取数据
train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])#划分训练集和验证集

# 输出数据大小.
print(f"""train_data size: {train_data.shape} 
valid_data size: {valid_data.shape} 
test_data size: {test_data.shape}""")

#选择特征，相当于重新制作x部分
x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])#选择是否所有的特征都需要使用

# 输出特征的数量，如果是所有：
print(f'number of features: {x_train.shape[1]}')

train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train),COVID19Dataset(x_valid, y_valid),COVID19Dataset(x_test)

# 用Pytorch中的DataLoader将上面获得数据集封装成一个一个batch的形式，即一个数据集由batch_size个batch组成.
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
"""
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False)
dataset：加载的数据集(Dataset对象)
batch_size：batch size
shuffle:：是否将数据打乱
sampler： 样本抽样
num_workers：使用多进程加载的进程数，0代表不使用多进程
collate_fn： 如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可
pin_memory：是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
drop_last：dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃
"""

class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: 修改网络架构，注意输入输出的维度.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),#一个输入为input_dim，输出为16的全连接层
            nn.ReLU(),#激活函数将非线性特性引入模型，激发隐藏节点以产生更理想的输出
            nn.Linear(16, 8),#一个输入为16，输出为8的全连接层
            nn.ReLU(),
            nn.Linear(8, 1)#输出1维，即为预测值
        )

    def forward(self, x):#模型向前传播
        x = self.layers(x)#输入数据，输出值
        x = x.squeeze(1) # (B, 1) -> (B)，去掉tensor中为1的维度，即对第二维进行压缩, 即把一整个batch的结果，压缩在tensor的一个维度里
        return x


def trainer(train_loader, valid_loader, model, config, device):  # 5个参数，训练集，验证集，待训练的网络，cpu/gpu

    criterion = nn.MSELoss(reduction='mean')  # 定义损失函数loss，均方误差，是预测值与真实值之差的平方和的平均值，作业中不要修改

    # 定义优化器.
    """
     TODO: 查看连接 https://pytorch.org/docs/stable/optim.html 学习更多可用的优化器
     TODO: L2 正则化 (优化器(设置weight decay...) 或者 自己定义一些超参数). 
     PS：L2 正则化设置weight decay权重衰减为解决过拟合
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)


    if not os.path.isdir('./models'):  # 判断./models是否为目录
        os.mkdir('./models')  # 不是的话要创建该目录

    n_epochs, best_loss, step, early_stop_count = config[
        'n_epochs'], math.inf, 0, 0  # 总共跑几轮，最好的loss值（初始为正无穷），当前batch的第几个数据，提前停止计数器

    # 训练loop：
    for epoch in range(n_epochs):  # 共n_epochs个循环
        model.train()  # 设置模型为训练模式，一些模块在训练和测试模式下不同（如BN，dropout）
        loss_record = []  # 设置list记录损失值

        # tqdm 模块可以可视化训练进度
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:  # 一个batch一个batch的看
            optimizer.zero_grad()  # 看每个batch的时候优化器中的梯度要重新设置为0
            x, y = x.to(device), y.to(device)  # 把训练数据x,y都加入gpu进行计算
            pred = model(x)  # 预测pred值，是一个tensor元组，有batchsize个一维tensor组成
            loss = criterion(pred, y)  # 计算loss值，一个batch的误差取平均
            loss.backward()  # 计算梯度，反向传播
            optimizer.step()  # 更新参数
            step += 1  # 数值=batch 的个数,累加步数，通常用于记录进行了多少次参数更新。
            loss_record.append(loss.detach().item())  # .detach（）：去掉计算梯度的部分，.item（）以数值的形式传回CPU上，加入list loss_record中
            # 显示当前epoch数值，并输出该batch下loss的数值
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')  # 进度条描述（开头）
            train_pbar.set_postfix({'loss=': loss.detach().item()})  # 进度条动态结果（结尾）

        mean_train_loss = sum(loss_record) / len(loss_record)  # 计算所有batch的loss 的平均
        writer.add_scalar('Loss/train', mean_train_loss,
                      step)  # 绘制可视化曲线，名称为Loss/train，纵坐标为loss值，横坐标为step值，step原来用在这里，牛逼

        model.eval()  # 设置模型为验证模式，即固定模型参数不会改变
        loss_record = []  # 初始化loss列表
        for x, y in valid_loader:  # 一个batch一个batch的看验证数据集
            x, y = x.to(device), y.to(device)  # 将x，y加入device中（gpu）
            with torch.no_grad():  # 不进行梯度计算
                pred = model(x)  # 做预测
                loss = criterion(pred, y)  # 计算验证loss

            loss_record.append(loss.item())  # 本身无梯度计算，则直接tensor转item到cpu中即可，无需像训练中使用detach（）

        mean_valid_loss = sum(loss_record) / len(loss_record)  # 计算所有batch（即一轮epoch）的平均loss值
        print(
            f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')  # 输出两个loss结果
        writer.add_scalar('Loss/valid', mean_valid_loss, step)  # 绘制可视化曲线

        # 若模型提前停止优化，则停止训练操作

        if mean_valid_loss < best_loss:  # 判断该轮验证loss是否为最小
            best_loss = mean_valid_loss  # 是则更新best_loss
            torch.save(model.state_dict(), config['save_path'])  # 存储当前最好模型的所有参数，
            print('Saving model with loss {:.3f}...'.format(best_loss))  # 当前最小的loss
            early_stop_count = 0
        else:
            early_stop_count += 1  # 验证loss不为最小则在提前停止计数器中记上一笔

        if early_stop_count >= config['early_stop']:  # 连续400个epoch 验证集上loss不为最小就提前停止训练
            print('\nModel is not improving, so we halt the training session.')


model = My_Model(input_dim=x_train.shape[1]).to(device) # 把模型和数据放在同一个device中
trainer(train_loader, valid_loader, model, config, device)# 调用训练函数

def predict(test_loader, model, device):
    '''用测试集做预测'''
    # 设置模型在验证模式下
    model.eval()
    #定义了一个空的列表list用于存储每个样本的预测结果
    preds = []
    #tqdm是Python 进度条库,可以显示预测测试集的进度
    for x in tqdm(test_loader):
        x = x.to(device)#将tensor加入指定的device上计算，后续都在这个device上计算
        with torch.no_grad():  #不进行梯度计算
            pred = model(x)  #输出预测结果
            preds.append(pred.detach().cpu())#阻断反传计算梯度，同时将数据移动到cpu中，pred是tensor格式,而preds是tensor的元组
    preds = torch.cat(preds, dim=0).numpy()#转为数组格式
    return preds

def save_pred(preds, file):
    ''' 保存测试结果到指定文件 '''
    with open(file, 'w') as fp: #使用 with 语句打开文件，文件模式为写入（'w'）。file 是文件的路径，fp 是文件对象。
        writer = csv.writer(fp) #创建一个 CSV 格式的写入对象，用于将数据写入文件。
        writer.writerow(['id', 'tested_positive'])#写入文件的第一行，包含两个列标题，分别是 'id' 和 'tested_positive'。
        for i, p in enumerate(preds):   #使用 enumerate 函数迭代 preds 列表中的预测结果。i 是索引，p 是预测值。
             writer.writerow([i, p]) #一行一行写入


model = My_Model(input_dim=x_train.shape[1]).to(device) # 把模型和数据放在同一个device中
model.load_state_dict(torch.load(config['save_path']))#将之前保存的模型参数赋值到新定义的模型中
preds = predict(test_loader, model, device) # 预测
save_pred(preds, 'pred.csv') #降结果保存在文件pred.csv中

