# 导入基础训练器类
from trainers.base_trainer import base_trainer
# 导入数据提供者函数
from data_providers.data_provider import data_provider
# 导入评估指标函数
from utils.metrics import masked_mae_np, masked_mse_np, masked_mape_np
# 导入PyTorch库
import torch
# 导入NumPy库
import numpy as np
# 导入操作系统相关功能
import os
# 导入进度条显示库
from tqdm import tqdm


class Trainer(base_trainer):
    """
    时序预测模型的训练器类，继承自base_trainer
    负责模型的训练、验证和测试过程
    """
    def __init__(self, configs):
        """
        初始化训练器
        
        参数:
            configs: 配置文件路径或配置对象
        """
        # 调用父类初始化方法
        super(Trainer, self).__init__(configs)
        # 设置随机种子，确保实验可重复性
        self._set_seed()
        # 获取日志记录器
        self._get_logger()
        # 获取计算设备（CPU或GPU）
        self.device = self._acquire_device()
        # 构建模型并移至指定设备
        self.model = self._build_model().to(self.device)
        # 创建Adam优化器
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.configs.learning_rate)
        # 获取训练数据集和数据加载器
        self.train_data, self.train_loader = self._get_data('train')
        # 获取验证数据集和数据加载器
        self.val_data, self.val_loader = self._get_data('val')
        # 获取测试数据集和数据加载器
        self.test_data, self.test_loader = self._get_data('test')
        # 如果需要恢复训练，则加载之前的模型
        if self.configs.resume:
            self._resume_()
        else:
            # 初始化最佳验证指标
            self.best_vali_metrics = {'mae': np.inf, 'mse': np.inf, 'mape': np.inf}
    
    def _resume_(self):
        """
        从检查点恢复模型训练
        """
        # 构建检查点文件路径
        path = os.path.join(self.configs.ckpts_path, self.configs.model.name, self.configs.dataset.name[0])
        path = os.path.join(path, 'checkpoint.pth')
        # 加载检查点
        ckpt = torch.load(path)
        # 加载模型参数
        self.model.load_state_dict(ckpt['model'])
        # 加载最佳验证指标
        self.best_vali_metrics = ckpt['best_metric']

    def _save_model(self):
        """
        保存模型检查点
        """
        # 构建保存路径
        path = os.path.join(self.configs.ckpts_path, self.configs.model.name, self.configs.dataset.name[0])
        # 如果路径不存在，则创建
        if not os.path.exists(path):
            os.makedirs(path)
        # 创建检查点字典，包含模型参数和最佳指标
        checkpoint = {
            'model': self.model.state_dict(),
            'best_metric': self.best_vali_metrics
        }
        # 保存检查点
        torch.save(checkpoint, os.path.join(path, f'checkpoint.pth'))
    
    def _get_data(self, flag):
        """
        获取指定类型的数据集和数据加载器
        
        参数:
            flag: 数据集类型，'train'、'val'或'test'
            
        返回:
            data_set: 数据集对象
            data_loader: 数据加载器对象
        """
        data_set, data_loader = data_provider(self.configs.dataset, flag)
        return data_set, data_loader

    def train(self):
        """
        训练模型的主函数
        """
        # 设置模型为训练模式
        self.model.train()
        # 循环训练指定的轮数
        for epoch in range(self.configs.epochs):
            # 遍历训练数据集中的每个批次
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(self.train_loader)):
                # 将输入数据移至指定设备
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                # 将时间标记数据转换为长整型并移至指定设备
                batch_x_mark = {key: value.long().to(self.device) for key, value in batch_x_mark.items()}
                batch_y_mark = {key: value.long().to(self.device) for key, value in batch_y_mark.items()}
                
                # 对输入和目标数据进行标准化处理
                batch_x = self.train_data.transform(batch_x)
                batch_y = self.train_data.transform(batch_y)

                # 清空梯度
                self.optim.zero_grad()
                # 前向传播，获取模型输出
                results = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                # 获取损失
                loss = results["loss"]
                # 反向传播
                loss.backward()
                # 更新模型参数
                self.optim.step()
        
            # 记录当前轮次信息
            self.logger.info("Epoch: {} ".format(epoch + 1))
            # 在验证集上评估模型
            vali_metrics = self.eval(self.val_loader)
            # 在测试集上评估模型
            test_metrics = self.eval(self.test_loader)

            # 记录验证集上的评估指标
            self.logger.info(f"On Valid Set, MAE:{vali_metrics['mae']}, MSE:{vali_metrics['mse']}, MAPE:{vali_metrics['mape']}")
            # 记录测试集上的评估指标
            self.logger.info(f"On Test Set, MAE:{test_metrics['mae']}, MSE:{test_metrics['mse']}, MAPE:{test_metrics['mape']}")

            # 如果当前模型在验证集上的MSE优于之前的最佳值，则保存模型
            if vali_metrics['mse'] < self.best_vali_metrics['mse']:
                self.best_vali_metrics = vali_metrics
                self._save_model()
                self.logger.info('best model saved')
    
    def eval(self, data_loader):
        """
        在指定数据加载器上评估模型
        
        参数:
            data_loader: 数据加载器对象
            
        返回:
            包含评估指标的字典
        """
        # 设置模型为评估模式
        self.model.eval()
        # 初始化预测值和真实值列表
        y_pred = []
        y_true = []

        # 禁用梯度计算，提高评估速度和减少内存使用
        with torch.no_grad():
            # 遍历数据集中的每个批次
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(data_loader)):
                # 将输入数据转换为浮点型并移至指定设备
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                # 将时间标记数据转换为长整型并移至指定设备
                batch_x_mark = {key: value.long().to(self.device) for key, value in batch_x_mark.items()}
                batch_y_mark = {key: value.long().to(self.device) for key, value in batch_y_mark.items()}

                # 对输入和目标数据进行标准化处理
                batch_x = self.train_data.transform(batch_x)
                batch_y = self.train_data.transform(batch_y)

                # 前向传播，获取模型输出
                result = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                # 获取预测值
                outputs = result['y_hat']
                # 收集预测值和真实值
                y_pred.append(outputs)
                y_true.append(batch_y)
            # 将所有批次的预测值和真实值拼接起来
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)
            # 计算平均绝对误差
            mae = masked_mae_np(y_true, y_pred, torch.nan)
            # 计算均方误差
            mse = masked_mse_np(y_true, y_pred, torch.nan)
            # 计算平均绝对百分比误差
            mape = masked_mape_np(y_true, y_pred, torch.nan)

        # 返回包含评估指标的字典
        return {'mae': mae, 'mse': mse, 'mape': mape}

                
