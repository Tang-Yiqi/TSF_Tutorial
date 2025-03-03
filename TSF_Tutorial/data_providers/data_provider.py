from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler, SequentialSampler
import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
from utils.read_cfg import read_cfg
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
from einops import rearrange

# 数据集字典，将数据集名称映射到对应的文件路径
data_dict = {
    'ETTh1': 'ETT-small/ETTh1.csv',
    'ETTh2': 'ETT-small/ETTh2.csv',
    'ETTm1': 'ETT-small/ETTm1.csv',
    'ETTm2': 'ETT-small/ETTm2.csv',
    'PEMS03': 'PEMS03/PEMS03.csv',
    'PEMS04': 'PEMS04/PEMS04.csv',
    'PEMS07': 'PEMS07/PEMS07.csv',
    'PEMS08': 'PEMS08/PEMS08.csv',
    'weather': 'weather/weather.csv',
    'traffic': 'traffic/traffic.csv',
    'electricity': 'electricity/electricity.csv',
    'illness': 'illness/national_illness.csv',
    }


# 添加周期批次采样器类
class CycleBatchSampler(BatchSampler):
    """
    基于周期的批次采样器，使得每个批次中的样本间隔为指定的周期
    
    参数:
        sampler: 基础采样器，通常是SequentialSampler
        cycle: 周期间隔
        batch_size: 批次大小
        drop_last: 是否丢弃最后一个不完整的批次
        
    示例:
        对于原始数据[0,1,2,3,4,5,6,7,8,9,10,11,12,...]
        如果cycle=4, batch_size=3, 则批次为:
        [0,4,8], [1,5,9], [2,6,10], [3,7,11], [12,16,20], ...
    """
    def __init__(self, sampler, cycle, batch_size, drop_last=False):
        # 调用父类构造函数
        super().__init__(sampler, batch_size, drop_last)
        
        # 成员变量初始化
        self.sampler = sampler        # 基础采样器
        self.cycle = cycle            # 周期间隔
        self.BatchSize = batch_size   # 批次大小
        self.DropLast = drop_last     # 是否丢弃最后一个不完整的批次
        self.SampleNum = len(sampler) # 样本总数
        
    def __iter__(self):
        """
        每次被调用时返回一个批次的索引列表
        
        yield与return的区别：
            yield：返回一个值，并暂停函数，下次调用时从上次暂停的地方继续执行
            return：返回一个值，并结束函数
        """
        
        for BatchCycleStart in range(0 , self.SampleNum , self.BatchSize * self.cycle):#batch采样的大循环：0 ， 12 ， 24
            for BatchStart in range(BatchCycleStart , BatchCycleStart + self.cycle - 1):#每个batch的开始：0 ， 1 ， 2 ， 3
                
                if self.DropLast and BatchStart + (self.BatchSize - 1) * self.cycle > self.SampleNum:#如果设置了drop_last且此时无法生成完整的batch
                    break
                #如果drop_last为False，则可能产生多个不完整的batch
                BatchIndex = []
                for i in range(self.BatchSize):#batch内的索引0 , 4 , 8
                    if BatchStart + i * self.cycle >= self.SampleNum:#如果索引超出样本总数
                        break
                    SampleIndex = BatchStart + i * self.cycle
                    BatchIndex.append(SampleIndex)
                yield BatchIndex
    
    def __len__(self):
        """
        返回批次总数
        """
        BatchNum = 0#批次总数
        
        BatchCycleNum = self.SampleNum // (self.BatchSize * self.cycle) #完整的大循环数
        LeftNum = self.SampleNum - BatchCycleNum * self.BatchSize * self.cycle #不能构成大循环的，剩余样本数
        
        BatchNum += BatchCycleNum * self.cycle #完整的大循环数可生成的batch数
        
        if self.DropLast:
            BatchNum += LeftNum - (self.BatchSize - 1) * self.cycle#按照每个batch的最后那个样本，计算剩余样本数
        else:
            BatchNum += min(LeftNum - 1, self.cycle)#按照每个batch的第一个样本，计算剩余样本数
            
        
        return BatchNum


def data_provider(args, flag='train', para=False, use_cycle_sampler=False):
    """
    数据提供者函数，创建数据集和数据加载器
    
    参数:
        args: 配置参数
        flag: 数据集类型，'train'、'val'或'test'
        para: 是否使用并行加载
        use_cycle_sampler: 是否使用周期采样器
        
    返回:
        dataset: 数据集对象
        dataloader: 数据加载器对象
    """
    # 如果是训练集，则打乱数据，否则不打乱
    shuffle = True if flag == 'train' else False
    # 从配置中获取批次大小
    batch_size = args.sync.batch_size
    # 创建自定义数据集
    dataset = CustomDataset(args, flag)
    
    # 根据是否使用周期采样器创建不同的数据加载器
    if use_cycle_sampler and 'cycle' in args.sync:
        # 使用周期批次采样器
        cycle = args.sync.cycle
        # 创建顺序采样器
        setequential_sampler = SequentialSampler(dataset)
        # 创建周期批次采样器
        batch_sampler = CycleBatchSampler(setequential_sampler, cycle, batch_size, drop_last=False)
        # 使用batch_sampler创建DataLoader
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler)
    else:
        # 使用原有的随机采样
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    return dataset, dataloader


# 添加一个便捷函数，用于创建基于周期的数据加载器
def cycle_data_provider(args, flag='train', para=False):
    """
    创建基于周期的数据加载器
    
    参数:
        args: 配置参数
        flag: 数据集类型，'train'、'val'或'test'
        para: 是否使用并行加载
        
    返回:
        dataset: 数据集对象
        dataloader: 数据加载器对象
    """
    return data_provider(args, flag, para, use_cycle_sampler=True)


class CustomDataset(Dataset):
    """
    自定义数据集类，用于加载和处理时序数据
    """
    def __init__(self, args, flag='train'):
        """
        初始化数据集
        
        参数:
            args: 配置参数
            flag: 数据集类型，'train'、'val'或'test'
        """
        # 保存配置参数
        self.args = args
        # 获取任务类型
        self.task = self.args.task
        # 获取是否进行数据缩放
        self.scale = self.args.scale
        # 创建标准化器
        self.scaler = StandardScaler()
        # 获取通道数
        self.n_channels = self.args.sync.n_channels
        # 如果是预测任务，则设置相关参数
        if self.task == 'forecasting':
            # 获取预测类型
            self.pred_type = self.args.pred_type
            # 获取历史长度（输入长度）
            self.his_len = self.args.sync.input_len
            # 获取预测长度（输出长度）
            self.pred_len = self.args.sync.output_len
        # 数据集类型映射
        set_types = {'train': 0, 'val': 1, 'test': 2}
        # 设置数据集类型
        self.set_type = set_types[flag]
        # 读取数据
        self.__read_data__()

    def __read_data__(self):
        """
        读取和预处理数据
        """
        # 读取CSV文件
        df_raw = pd.read_csv(self.args.root_dir + data_dict[self.args.name[0]])
        # 对ETT数据集进行特殊处理
        if self.args.name[0] == 'ETTh1' or self.args.name[0] == 'ETTh2' or self.args.name[0] == 'ETTm1' or self.args.name[0] == 'ETTm2':
            # 将日期列转换为datetime类型
            df_raw['date'] = pd.to_datetime(df_raw['date'])
            # 提取日期中的天数
            df_raw['day'] = df_raw.date.apply(lambda row: row.day, 1)
            # 注释掉的数据过滤代码
            # df_raw = df_raw.drop(df_raw[df_raw['day'] == 31].index)
            # df_raw = df_raw.drop(df_raw[(df_raw['date'] >= '2017-07-23 21:00') & (df_raw['date'] <= '2017-07-29 17:00')].index)
            # 删除day列
            df_raw = df_raw.drop(labels='day', axis=1)
        # 获取所有列名
        cols = list(df_raw.columns)
        # 根据预测类型选择数据
        if self.pred_type == 'm2m' or self.pred_type == 'm2u':
            # 多变量到多变量或多变量到单变量，选择除第一列（日期）外的所有列
            df_data = df_raw[cols[1:]]
        elif self.pred_type == 'u2u':
            # 单变量到单变量，只选择最后一列
            df_data = df_raw[cols[-1]]

        # 根据数据集设置数据长度
        if self.args.name[0] == 'ETTh1' or self.args.name[0] == 'ETTh2':
            # ETTh数据集，每小时一个样本，20个月 * 30天 * 24小时
            data_length = 20 * 30 * 24
        elif self.args.name[0] == 'ETTm1' or self.args.name[0] == 'ETTm2':
            # ETTm数据集，每15分钟一个样本，20个月 * 30天 * 24小时 * 4（每小时4个样本）
            data_length = 20 * 30 * 24 * 4
        else:
            # 其他数据集，使用数据的实际长度
            data_length = len(df_data)
        # 对PEMS数据集进行特殊处理，填充缺失值
        if "PEMS" in self.args.name[0]:
            df_data = df_data.fillna(0)
        # 计算训练集、验证集和测试集的大小
        num_train = int(data_length * self.args.train_ratio)
        num_val = int(data_length * self.args.val_ratio)
        num_test = int(data_length * self.args.test_ratio)

        # 设置数据边界
        border1s = [0, num_train - self.his_len, num_train + num_val - self.his_len]
        border2s = [num_train, num_train + num_val, data_length]

        # 根据数据集类型选择边界
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # 获取训练数据，用于拟合标准化器
        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        # 保存均值和标准差，用于后续的标准化和反标准化
        self.means = self.scaler.mean_
        self.stds = self.scaler.scale_

        # 获取所有数据
        data = df_data.values
        # 获取时间戳数据
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        # 提取时间特征
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1) - 1  # 月份，从0开始
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1) - 1  # 日期，从0开始
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)  # 星期几，0-6
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)  # 小时，0-23
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute / 15, 1)  # 分钟，转换为15分钟间隔的索引
        # 删除date列，只保留提取的时间特征
        data_stamp = df_stamp.drop(labels='date', axis=1).values

        # 将数据转换为张量，并移至指定设备
        data = torch.tensor(data[border1:border2], device=self.args.sync.device)
        # 如果是训练集且需要缩放，则进行标准化
        # if self.set_type == 0 and self.scale:
        #     data = (data - self.means) / self.stds
        
        # 设置输入和输出数据（在时序预测中，它们是同一个数据的不同时间段）
        self.data_x = data
        self.data_y = data

        # 保存时间戳数据
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
        获取数据集中的一个样本
        
        参数:
            index: 样本索引
            
        返回:
            x: 输入数据
            y: 目标数据
            x_mark: 输入数据的时间标记
            y_mark: 目标数据的时间标记
        """
        # 初始化时间标记字典
        x_mark, y_mark = {}, {}
        # 调整索引，考虑历史长度
        index = index + self.his_len
        # 获取输入数据，形状为[通道数, 历史长度]
        x = self.data_x[index - self.his_len:index, ...].transpose(-1, -2).float()
        # 获取目标数据，形状为[通道数, 预测长度]
        y = self.data_y[index:index + self.pred_len, ...].transpose(-1, -2).float()
        # 设置输入数据的时间标记
        x_mark['time_stamp'] = self.data_stamp[index - self.his_len:index]
        x_mark['pos_stamp'] = torch.arange(0, self.his_len)
        # 设置目标数据的时间标记
        y_mark['time_stamp'] = self.data_stamp[index:index + self.pred_len]
        y_mark['pos_stamp'] = torch.arange(0, self.pred_len)

        # 设置通道索引
        x_mark['channel'] = torch.arange(0, self.n_channels)
        y_mark['channel'] = torch.arange(0, self.n_channels)
        return x, y, x_mark, y_mark

    def __len__(self):
        """
        获取数据集长度
        
        返回:
            数据集中的样本数量
        """
        # 数据总长度减去历史长度和预测长度，再加1
        return self.data_x.shape[0] - self.his_len - self.pred_len + 1

    def transform(self, data):
        """
        对数据进行标准化处理
        
        参数:
            data: 输入数据，形状为[批次大小, 通道数, 序列长度]
            
        返回:
            标准化后的数据
        """
        # 获取数据形状
        B, C, L = data.shape
        # 重排数据形状为[批次大小*序列长度, 通道数]
        data = rearrange(data, 'b c l -> (b l) c')
        # 使用均值和标准差进行标准化
        out = (data - torch.Tensor(self.means).to(data)) / torch.Tensor(self.stds).to(data)
        # 将数据形状恢复为[批次大小, 通道数, 序列长度]
        out = rearrange(out, '(b l) c -> b c l', l=L)
        return out

    def inverse_transform(self, data):
        """
        对标准化的数据进行反标准化处理
        
        参数:
            data: 标准化后的数据，形状为[批次大小, 通道数, 序列长度]
            
        返回:
            反标准化后的数据
        """
        # 获取数据形状
        B, C, L = data.shape
        # 重排数据形状为[批次大小*序列长度, 通道数]
        data = rearrange(data, 'b c l -> (b l) c')
        # 使用均值和标准差进行反标准化
        out = data * (torch.Tensor(self.stds).to(data)) + torch.Tensor(self.means).to(data)
        # 将数据形状恢复为[批次大小, 通道数, 序列长度]
        out = rearrange(out, '(b l) c -> b c l', l=L)
        return out

