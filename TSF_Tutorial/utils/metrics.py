# 导入NumPy库
import numpy as np
# 导入PyTorch库
import torch
# 导入快速动态时间规整算法
from fastdtw import fastdtw
# 导入进度条显示库
from tqdm import tqdm

def mask_np(array, null_val):
    """
    创建掩码，标记非空值
    
    参数:
        array: 输入数组
        null_val: 空值标记
        
    返回:
        掩码张量，非空值为1.0，空值为0.0
    """
    return torch.not_equal(array, null_val).float()

def masked_mape_np(y_true, y_pred, null_val=torch.nan, reduction='mean'):
    """
    计算带掩码的平均绝对百分比误差（MAPE）
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        null_val: 空值标记，默认为NaN
        reduction: 归约方式，'mean'、'sum'或'none'
        
    返回:
        MAPE值（百分比）
    """
    # 创建掩码，标记非空值
    mask = mask_np(y_true, null_val)
    # 归一化掩码，使其平均值为1
    mask /= mask.mean()
    # 计算绝对百分比误差
    mape = torch.abs((y_pred - y_true) / y_true)
    # 应用掩码
    mape = mask * mape
    # 根据归约方式返回结果
    if reduction == 'mean':
        return torch.mean(mape) * 100
    elif reduction == 'sum':
        return torch.sum(mape) * 100
    elif reduction == 'none':
        return mape * 100
    else:
        raise ValueError('reduction should be mean, sum or none')


def masked_rmse_np(y_true, y_pred, null_val=torch.nan):
    """
    计算带掩码的均方根误差（RMSE）
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        null_val: 空值标记，默认为NaN
        
    返回:
        RMSE值
    """
    # 创建掩码，标记非空值
    mask = mask_np(y_true, null_val)
    # 归一化掩码，使其平均值为1
    mask /= mask.mean()
    # 计算均方误差
    mse = (y_true - y_pred) ** 2
    # 返回均方根误差
    return torch.sqrt(torch.mean(mask * mse))


def masked_mse_np(y_true, y_pred, null_val=torch.nan):
    """
    计算带掩码的均方误差（MSE）
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        null_val: 空值标记，默认为NaN
        
    返回:
        MSE值
    """
    # 创建掩码，标记非空值
    mask = mask_np(y_true, null_val)
    # 归一化掩码，使其平均值为1
    mask /= mask.mean()
    # 计算均方误差
    mse = (y_true - y_pred) ** 2
    # 返回带掩码的均方误差
    return torch.mean(mask * mse)


def masked_mae_np(y_true, y_pred, null_val=torch.nan):
    """
    计算带掩码的平均绝对误差（MAE）
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        null_val: 空值标记，默认为NaN
        
    返回:
        MAE值
    """
    # 创建掩码，标记非空值
    mask = mask_np(y_true, null_val)
    # 归一化掩码，使其平均值为1
    mask /= mask.mean()
    # 计算绝对误差
    mae = torch.abs(y_true - y_pred)
    # 返回带掩码的平均绝对误差
    return torch.mean(mask * mae)

def get_dtw(y_true, y_pred, null_val=torch.nan):
    """
    计算动态时间规整（DTW）距离
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        null_val: 空值标记，默认为NaN
        
    返回:
        平均DTW距离
    """
    # 重塑张量形状
    y_true = y_true.reshape(y_true.shape[-1], y_true.shape[-2])
    y_pred = y_pred.reshape(y_pred.shape[-1], y_pred.shape[-2])
    # 初始化DTW距离列表
    dtw_list = []
    # 定义曼哈顿距离函数
    manhattan_distance = lambda x, y: np.abs(x - y)
    # 对每个通道计算DTW距离
    for i in range(y_pred.shape[0]):
        x = y_pred[i].reshape(-1, 1)
        y = y_true[i].reshape(-1, 1)
        # 使用fastdtw计算DTW距离
        d, _, = fastdtw(x, y, dist=manhattan_distance)
        dtw_list.append(d)
    # 计算平均DTW距离
    dtw = np.array(dtw_list).mean()
    return dtw

