import os
from torchvision import transforms

class Config:
    def __init__(self):
        # 数据集配置
        self.data_root = 'data/VOC2012_trainval/VOCdevkit/VOC2012'
        self.num_classes = 21  # VOC2012有20个类别+背景
        
        # 训练配置
        self.batch_size = 2  # 降低批次大小
        self.num_epochs = 50  # 减少训练轮数
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.num_workers = 2  # 减少数据加载的工作进程数
        self.pretrained = True
        
        # 模型配置
        self.input_size = (256, 256)  # 降低图像分辨率
        
        # 路径配置
        self.checkpoint_dir = 'checkpoints'
        self.log_dir = 'logs'
        
        # 数据增强
        self.train_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 保存和日志配置
        self.save_freq = 5  # 每多少个epoch保存一次检查点
        self.visualize = True  # 是否可视化验证结果
        
        # 模型配置
        self.backbone = 'resnet38'
        self.fusion_channels = 512
        self.attention_channels = 256 