import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # 数据集配置
    dataset_name: str = 'voc'  # 'voc' 或 'sbd'
    dataset_root: str = 'data'
    train_split: str = 'train'
    val_split: str = 'val'
    
    # 模型配置
    backbone: str = 'resnet38'
    pretrained: bool = True
    num_classes: int = 21  # VOC2012的类别数
    
    # 训练配置
    batch_size: int = 8
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 100
    
    # 数据增强配置
    image_size: tuple = (512, 512)
    random_rotation: float = 10.0
    color_jitter: tuple = (0.2, 0.2, 0.2, 0.1)
    
    # 路径配置
    log_dir: str = 'logs'
    checkpoint_dir: str = 'checkpoints'
    resume_path: Optional[str] = None
    
    def __post_init__(self):
        """创建必要的目录"""
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 设置数据集路径
        if self.dataset_name.lower() == 'voc':
            self.dataset_root = os.path.join(self.dataset_root, 'VOC2012')
        elif self.dataset_name.lower() == 'sbd':
            self.dataset_root = os.path.join(self.dataset_root, 'SBD')
        else:
            raise ValueError(f'不支持的数据集: {self.dataset_name}') 