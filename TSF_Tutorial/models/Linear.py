# 导入PyTorch库
import torch
# 导入PyTorch神经网络模块
import torch.nn as nn
# 导入PyTorch函数式接口
import torch.nn.functional as F

class Model(nn.Module):
    """
    线性模型类，用于时序预测
    
    这是一个简单的线性模型，将输入序列通过线性层映射到输出序列
    模型会对输入数据进行标准化处理，然后应用线性变换，最后反标准化得到预测结果
    """
    def __init__(self, input_len, output_len, num_channels, configs):
        """
        初始化线性模型
        
        参数:
            input_len: 输入序列长度
            output_len: 输出序列长度
            num_channels: 通道数（特征维度）
            configs: 模型配置参数
        """
        # 调用父类初始化方法
        super(Model, self).__init__()
        # 保存输入序列长度
        self.input_len = input_len
        # 保存输出序列长度
        self.output_len = output_len
        # 保存通道数
        self.num_channels = num_channels
        # 创建线性层，将输入序列映射到输出序列
        self.linear = nn.Linear(input_len, output_len)

    def forward(self, x, y, x_mark, y_mark):
        """
        前向传播函数
        
        参数:
            x: 输入数据，形状为[批次大小, 通道数, 输入序列长度]
            y: 目标数据，形状为[批次大小, 通道数, 输出序列长度]
            x_mark: 输入数据的时间标记
            y_mark: 目标数据的时间标记
            
        返回:
            包含预测结果和损失的字典
        """
        # 计算输入数据在时间维度上的均值，保持维度
        x_means = x.mean(dim=-1, keepdim=True)
        # 计算输入数据在时间维度上的标准差，保持维度，并添加小值防止除零
        x_stds = x.std(dim=-1, keepdim=True) + 1e-5
        # 对输入数据进行标准化
        x = (x - x_means) / x_stds
        # 应用线性变换，将输入序列映射到输出序列
        y_hat = self.linear(x)
        # 对预测结果进行反标准化，恢复到原始数据的尺度
        y_hat = y_hat * x_stds + x_means

        # 创建结果字典
        result = {}
        # 保存预测结果
        result['y_hat'] = y_hat
        # 计算均方误差损失
        result['loss'] = F.mse_loss(y_hat, y)
        return result

