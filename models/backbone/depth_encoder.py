import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(DepthEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 深度特征提取块
        self.depth_blocks = nn.ModuleList([
            DepthBlock(out_channels, out_channels * 2),
            DepthBlock(out_channels * 2, out_channels * 4),
            DepthBlock(out_channels * 4, out_channels * 8),
            DepthBlock(out_channels * 8, out_channels * 16)
        ])
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 初始特征提取
        features = [self.conv1(x)]
        
        # 深度特征提取
        for block in self.depth_blocks:
            x = block(x)
            features.append(x)
        
        return features

class DepthBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 深度特定的注意力模块
        self.attention = DepthAttention(out_channels)
        
        # 下采样
        self.downsample = nn.MaxPool2d(2)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征提取
        out = self.conv1(x)
        out = self.conv2(out)
        
        # 应用注意力
        out = self.attention(out)
        
        # 下采样
        out = self.downsample(out)
        
        return out

class DepthAttention(nn.Module):
    def __init__(self, channels):
        super(DepthAttention, self).__init__()
        self.channels = channels
        
        # 深度梯度注意力
        self.gradient_attention = nn.Sequential(
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 深度范围注意力
        self.range_attention = nn.Sequential(
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 计算梯度注意力
        gradient_weights = self.gradient_attention(x)
        
        # 计算范围注意力
        range_weights = self.range_attention(x)
        
        # 组合注意力
        attention_weights = gradient_weights * range_weights
        
        # 应用注意力
        out = x * attention_weights
        
        return out 