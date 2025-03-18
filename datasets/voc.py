import os
import numpy as np
from .base_dataset import BaseDataset
from PIL import Image

class VOC2012Dataset(BaseDataset):
    def __init__(self, root_dir, split='train', transform=None, target_transform=None):
        super(VOC2012Dataset, self).__init__(root_dir, transform, target_transform)
        self.split = split
        self.root_dir = root_dir
        
        # VOC2012的类别
        self.class_names = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]
        
        # 类别颜色映射
        self.colormap = self._create_colormap()
        
        # 加载数据集
        self._load_dataset()
    
    def _load_dataset(self):
        """加载数据集"""
        # 读取数据集分割文件
        split_file = os.path.join(self.root_dir, 'ImageSets', 'Segmentation', f'{self.split}.txt')
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        # 构建图像和掩码路径
        self.images = [
            os.path.join(self.root_dir, 'JPEGImages', f'{image_id}.jpg')
            for image_id in self.image_ids
        ]
        self.masks = [
            os.path.join(self.root_dir, 'SegmentationClass', f'{image_id}.png')
            for image_id in self.image_ids
        ]
    
    def _create_colormap(self):
        """创建类别颜色映射"""
        colormap = np.zeros((256, 3), dtype=np.uint8)
        colormap[0] = [0, 0, 0]  # 背景
        colormap[1] = [128, 0, 0]  # aeroplane
        colormap[2] = [0, 128, 0]  # bicycle
        colormap[3] = [128, 128, 0]  # bird
        colormap[4] = [0, 0, 128]  # boat
        colormap[5] = [128, 0, 128]  # bottle
        colormap[6] = [0, 128, 128]  # bus
        colormap[7] = [128, 128, 128]  # car
        colormap[8] = [64, 0, 0]  # cat
        colormap[9] = [192, 0, 0]  # chair
        colormap[10] = [64, 128, 0]  # cow
        colormap[11] = [192, 128, 0]  # diningtable
        colormap[12] = [64, 0, 128]  # dog
        colormap[13] = [192, 0, 128]  # horse
        colormap[14] = [64, 128, 128]  # motorbike
        colormap[15] = [192, 128, 128]  # person
        colormap[16] = [0, 64, 0]  # pottedplant
        colormap[17] = [128, 64, 0]  # sheep
        colormap[18] = [0, 192, 0]  # sofa
        colormap[19] = [128, 192, 0]  # train
        colormap[20] = [0, 64, 128]  # tvmonitor
        return colormap
    
    def get_colormap(self):
        return self.colormap
    
    def get_class_names(self):
        return self.class_names
    
    def get_class_weights(self):
        """计算类别权重"""
        # 统计每个类别的像素数量
        class_counts = np.zeros(len(self.class_names))
        for mask_path in self.masks:
            mask = np.array(Image.open(mask_path))
            for i in range(len(self.class_names)):
                class_counts[i] += np.sum(mask == i)
        
        # 计算权重
        total = np.sum(class_counts)
        weights = total / (len(self.class_names) * class_counts)
        weights[0] = 0  # 背景权重设为0
        
        return weights 