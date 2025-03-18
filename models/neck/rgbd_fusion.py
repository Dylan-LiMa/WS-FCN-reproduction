import torch
import torch.nn as nn
import torch.nn.functional as F

class RGBDFusion(nn.Module):
    def __init__(self, rgb_channels, depth_channels, out_channels):
        super(RGBDFusion, self).__init__()
        self.rgb_channels = rgb_channels
        self.depth_channels = depth_channels
        self.out_channels = out_channels
        
        # RGB特征处理
        self.rgb_conv = nn.Sequential(
            nn.Conv2d(rgb_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 深度特征处理
        self.depth_conv = nn.Sequential(
            nn.Conv2d(depth_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 注意力模块
        self.attention = CrossModalAttention(out_channels)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb_features, depth_features):
        # 处理RGB特征
        rgb_out = self.rgb_conv(rgb_features)
        
        # 处理深度特征
        depth_out = self.depth_conv(depth_features)
        
        # 特征融合
        fused_features = torch.cat([rgb_out, depth_out], dim=1)
        fused_features = self.fusion_conv(fused_features)
        
        # 应用注意力机制
        out = self.attention(fused_features)
        
        return out

class CrossModalAttention(nn.Module):
    def __init__(self, channels):
        super(CrossModalAttention, self).__init__()
        self.channels = channels
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
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
        # 空间注意力
        spatial_weights = self.spatial_attention(x)
        x = x * spatial_weights
        
        # 通道注意力
        channel_weights = self.channel_attention(x)
        x = x * channel_weights
        
        return x 