import torch
import torchvision.transforms.functional as F
import random
import numpy as np

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, rgb, depth, mask):
        for t in self.transforms:
            rgb, depth, mask = t(rgb, depth, mask)
        return rgb, depth, mask

class Resize:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, rgb, depth, mask):
        rgb = F.resize(rgb, self.size, F.InterpolationMode.BILINEAR)
        depth = F.resize(depth, self.size, F.InterpolationMode.NEAREST)
        mask = F.resize(mask, self.size, F.InterpolationMode.NEAREST)
        return rgb, depth, mask

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, rgb, depth, mask):
        if random.random() < self.p:
            rgb = F.hflip(rgb)
            depth = F.hflip(depth)
            mask = F.hflip(mask)
        return rgb, depth, mask

class RandomRotation:
    def __init__(self, degrees=10):
        self.degrees = degrees
    
    def __call__(self, rgb, depth, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        rgb = F.rotate(rgb, angle, F.InterpolationMode.BILINEAR)
        depth = F.rotate(depth, angle, F.InterpolationMode.NEAREST)
        mask = F.rotate(mask, angle, F.InterpolationMode.NEAREST)
        return rgb, depth, mask

class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, rgb, depth, mask):
        rgb = F.color_jitter(
            rgb,
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue
        )
        return rgb, depth, mask

class DepthNormalize:
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
    
    def __call__(self, rgb, depth, mask):
        # 将深度图转换为tensor并归一化
        depth = F.to_tensor(depth)
        depth = (depth - self.mean) / self.std
        return rgb, depth, mask

class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, rgb, depth, mask):
        # 将RGB图像转换为tensor并归一化
        rgb = F.to_tensor(rgb)
        rgb = F.normalize(rgb, self.mean, self.std)
        return rgb, depth, mask

def get_train_transforms(size):
    """获取训练数据转换
    
    Args:
        size: 目标图像大小
        
    Returns:
        Compose: 数据转换组合
    """
    return Compose([
        Resize(size),
        RandomHorizontalFlip(),
        RandomRotation(),
        ColorJitter(),
        DepthNormalize(),
        Normalize()
    ])

def get_val_transforms(size):
    """获取验证数据转换
    
    Args:
        size: 目标图像大小
        
    Returns:
        Compose: 数据转换组合
    """
    return Compose([
        Resize(size),
        DepthNormalize(),
        Normalize()
    ]) 