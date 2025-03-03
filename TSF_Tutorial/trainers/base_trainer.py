# 导入操作系统相关功能
import os
# 导入Linear模型
from models import Linear
# 导入PyTorch库
import torch
# 导入PyTorch CUDA后端
from torch.backends import cudnn
# 导入配置文件读取函数
from utils.read_cfg import read_cfg
# 导入OmegaConf库，用于处理配置
from omegaconf import OmegaConf
# 导入NumPy库
import numpy as np
# 导入random库
import random
# 导入参数解析库
import argparse
# 导入日志记录库
import logging
# 导入时间处理库
import time

# 模型字典，将模型名称映射到模型类
model_dict = {
    'Linear': Linear,
}

class base_trainer(object):
    """
    基础训练器类，为所有训练器提供通用功能
    """
    def __init__(self, args_path):
        """
        初始化基础训练器
        
        参数:
            args_path: 配置文件路径
        """
        # 读取基础配置文件
        args_base = read_cfg(args_path)
        # 构建数据集配置文件路径
        datasets_args_path = os.path.join(args_base.config_path, 'datasets', args_base.dataset + '.yaml')
        # 检查数据集配置文件是否存在
        if not os.path.exists(datasets_args_path):
            raise FileNotFoundError('Dataset config file not found.')
        # 构建模型配置文件路径
        model_args_path = os.path.join(args_base.config_path, 'models', args_base.model + '.yaml')
        # 检查模型配置文件是否存在
        if not os.path.exists(model_args_path):
            raise FileNotFoundError('Model config file not found.')
        # 构建任务配置文件路径
        task_args_path = os.path.join(args_base.config_path, 'tasks', args_base.task + '.yaml')
        # 检查任务配置文件是否存在
        if not os.path.exists(task_args_path):
            raise FileNotFoundError('Task config file not found.')

        # 读取各个配置文件
        args_dataset = read_cfg(datasets_args_path)
        args_model = read_cfg(model_args_path)
        args_task = read_cfg(task_args_path)

        # 合并同步配置
        args_sync = OmegaConf.merge(args_dataset.sync, args_task.sync, args_base.sync)
        args_base.sync = args_sync
        args_dataset.sync = args_sync
        args_model.sync = args_sync
        args_task.sync = args_sync

        # 保存更新后的配置文件
        OmegaConf.save(args_base, args_path)
        OmegaConf.save(args_dataset, datasets_args_path)
        OmegaConf.save(args_model, model_args_path)
        OmegaConf.save(args_task, task_args_path)
        
        # 处理辅助模型配置（如果存在）
        if 'aux_models' in args_model:
            for aux_model in args_model.aux_models:
                aux_model_args_path = os.path.join(args_base.config_path, 'models', aux_model + '.yaml')
                if not os.path.exists(aux_model_args_path):
                    raise FileNotFoundError('Model config file not found.')
                args_aux_model = read_cfg(aux_model_args_path)
                args_aux_model.sync = args_sync
                OmegaConf.save(args_aux_model, aux_model_args_path)

        # 将所有配置整合到self.configs中
        self.configs = args_base
        self.configs.dataset = args_dataset
        self.configs.model = args_model
        self.configs.task = args_task
        # 设置随机种子
        self._set_seed()

    def _set_seed(self):
        """
        设置随机种子，确保实验可重复性
        """
        # 设置NumPy随机种子
        np.random.seed(self.configs.sync.seed)
        # 设置Python随机种子
        random.seed(self.configs.sync.seed)
        # 设置PyTorch随机种子
        torch.manual_seed(self.configs.sync.seed)

        # 注释掉的CUDNN设置
        # cudnn.deterministic = True
        # cudnn.benchmark = True

        # 启用CUDA的Flash SDP（Scaled Dot Product）加速
        torch.backends.cuda.enable_flash_sdp(enabled=True)

    def _get_logger(self):
        """
        创建日志记录器
        """
        # 设置日志格式
        LOG_FORMAT = "%(asctime)s  %(message)s"
        # 设置日期格式
        DATE_FORMAT = "%m/%d %H:%M"

        # 创建控制台处理器
        console_handler = logging.StreamHandler()  # 输出到控制台
        # 配置基础日志设置
        logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
        # 获取日志记录器
        self.logger = logging.getLogger(__name__)   
        # 设置matplotlib日志级别为WARNING，减少不必要的输出
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        # 注释掉的处理器添加
        # self.logger.addHandler(console_handler)

        # 如果指定了日志路径，则创建文件处理器
        if self.configs.log_path is not None:
            # 获取日志目录
            log_path = os.path.dirname(self.configs.log_path)
            # 如果目录不存在，则创建
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            # 构建日志文件名，包含时间戳、数据集名称和模型名称
            log_name = self.configs.log_path + time.strftime(
                '%Y-%m-%d-%H-%M-%S') + f'_{self.configs.dataset.name[0]}_{self.configs.model.name}_.log'

            # 创建文件处理器
            file_handler = logging.FileHandler(log_name)  # 输出到文件
            # 将文件处理器添加到日志记录器
            self.logger.addHandler(file_handler)
            # 记录配置信息
            self.logger.info(self.configs)
            self.logger.info(self.configs.model)
            self.logger.info(self.configs.dataset)

    def _build_model(self):
        """
        构建模型
        
        返回:
            构建好的模型实例
        """
        # 根据配置中的模型名称从模型字典中获取对应的模型类，并创建模型实例
        model = model_dict[self.configs.model.name].Model(input_len = self.configs.sync.input_len,
                                                          output_len = self.configs.sync.output_len,
                                                          num_channels = self.configs.sync.n_channels,
                                                          configs = self.configs.model)
        return model

    def _acquire_device(self):
        """
        获取计算设备（CPU或GPU）
        
        返回:
            设备对象
        """
        # 从配置中获取设备信息
        device = self.configs.sync.device
        return device

    def _get_data(self, flag):
        """
        获取数据的抽象方法，需要在子类中实现
        
        参数:
            flag: 数据集类型
        """
        pass

    def _sava_model(self):
        """
        保存模型的抽象方法，需要在子类中实现
        注意：方法名有拼写错误，应为_save_model
        """
        pass
