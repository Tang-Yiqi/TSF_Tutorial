# 导入OmegaConf库，用于处理配置文件
from omegaconf import OmegaConf

def read_cfg(cfgdir):
    """
    读取配置文件
    
    参数:
        cfgdir: 配置文件路径
        
    返回:
        解析后的配置对象
    """
    # 使用OmegaConf加载配置文件
    args = OmegaConf.load(cfgdir)
    return args