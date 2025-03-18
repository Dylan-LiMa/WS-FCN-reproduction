import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET

class VOCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'JPEGImages')
        self.annotations_dir = os.path.join(root_dir, 'Annotations')
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        ann_path = os.path.join(self.annotations_dir, os.path.splitext(img_name)[0] + '.xml')
        
        # 读取图像
        image = Image.open(img_path).convert('RGB')
        
        # 解析XML标注
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        # 这里需要根据具体任务实现数据加载和预处理
        # 返回处理后的图像和标注
        return image, root 