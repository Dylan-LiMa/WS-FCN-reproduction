import torch
import torch.nn as nn
import torch.nn.functional as F

class FCA(nn.Module):
    """全局上下文注意力模块"""
    def __init__(self, in_channels, reduction=16):
        super(FCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SF2(nn.Module):
    """局部细节增强模块"""
    def __init__(self, in_channels):
        super(SF2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class WSFCN(nn.Module):
    """WS-FCN模型"""
    def __init__(self, num_classes, pretrained=False):
        super(WSFCN, self).__init__()
        # 基础特征提取
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # FCA模块
        self.fca = FCA(256)
        
        # SF2模块
        self.sf2 = SF2(256)
        
        # 分类头
        self.classifier = nn.Conv2d(256, num_classes, 1)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # 基础特征提取
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        # 应用FCA
        x = self.fca(x)
        
        # 应用SF2
        x = self.sf2(x)
        
        # 上采样到原始大小
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)
        
        # 分类
        x = self.classifier(x)
        
        return x 