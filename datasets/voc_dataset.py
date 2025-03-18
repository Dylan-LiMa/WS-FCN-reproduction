import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class VOCDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """VOC数据集加载器
        
        Args:
            root_dir: VOC数据集根目录
            split: 数据集划分 ['train', 'val']
            transform: 数据增强
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # 读取数据集划分文件
        split_file = os.path.join(root_dir, 'ImageSets', 'Segmentation', f'{split}.txt')
        with open(split_file, 'r') as f:
            self.images = [line.strip() for line in f.readlines()]
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 读取图像和标签
        image_name = self.images[idx]
        image_path = os.path.join(self.root_dir, 'JPEGImages', f'{image_name}.jpg')
        mask_path = os.path.join(self.root_dir, 'SegmentationClass', f'{image_name}.png')
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # 应用数据增强
        if self.transform:
            # 对图像进行变换
            image = self.transform(image)
            
            # 对标签进行resize，使用最近邻插值
            mask = mask.resize((256, 256), Image.NEAREST)
            mask = torch.from_numpy(np.array(mask)).long()
        
        return image, mask

    def get_image_size(self, idx):
        """获取图像尺寸
        
        Args:
            idx: 样本索引
            
        Returns:
            tuple: (width, height)
        """
        with Image.open(self.images[idx]) as img:
            return img.size 