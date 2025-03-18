import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        """初始化数据集
        
        Args:
            root_dir: 数据集根目录
            transform: 数据转换函数
            is_train: 是否为训练集
        """
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        # 图像和标签路径列表
        self.images = []
        self.depths = []
        self.masks = []
        
        # 类别名称和颜色映射
        self.class_names = []
        self.color_map = {}
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.images)
    
    def __getitem__(self, idx):
        """获取数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            tuple: (rgb_image, depth_image, mask)
        """
        # 加载RGB图像
        rgb_path = self.images[idx]
        rgb = Image.open(rgb_path).convert('RGB')
        
        # 加载深度图像
        depth_path = self.depths[idx]
        depth = Image.open(depth_path).convert('L')
        
        # 加载分割标签
        mask_path = self.masks[idx]
        mask = Image.open(mask_path)
        
        # 应用数据转换
        if self.transform:
            rgb, depth, mask = self.transform(rgb, depth, mask)
        
        return rgb, depth, mask
    
    def get_color_map(self):
        """获取类别颜色映射"""
        return self.color_map
    
    def get_class_names(self):
        """获取类别名称列表"""
        return self.class_names
    
    def get_class_weights(self):
        """计算类别权重"""
        # 统计每个类别的像素数量
        class_counts = np.zeros(len(self.class_names))
        total_pixels = 0
        
        for mask_path in self.masks:
            mask = np.array(Image.open(mask_path))
            for i in range(len(self.class_names)):
                class_counts[i] += np.sum(mask == i)
            total_pixels += mask.size
        
        # 计算权重
        weights = total_pixels / (len(self.class_names) * class_counts)
        weights[0] = 0  # 背景类权重设为0
        
        return weights 