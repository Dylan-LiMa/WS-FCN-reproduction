import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureFusion(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FeatureFusion, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        
        # 为每个输入特征创建1x1卷积层
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # 最终融合后的1x1卷积
        self.final_conv = nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=1)
        
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, features):
        # 确保输入特征列表长度与预期一致
        assert len(features) == len(self.in_channels_list), \
            f"Expected {len(self.in_channels_list)} features, got {len(features)}"
        
        # 处理每个输入特征
        processed_features = []
        for i, (feature, conv) in enumerate(zip(features, self.conv_layers)):
            # 1x1 卷积降维
            x = conv(feature)
            
            # 上采样到相同大小
            if i > 0:  # 第一个特征不需要上采样
                x = F.interpolate(x, size=features[0].shape[2:], mode='bilinear', align_corners=True)
            
            processed_features.append(x)
        
        # 特征融合
        x = torch.cat(processed_features, dim=1)
        x = self.final_conv(x)
        
        return x

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, kernel_size=1)
        
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 空间注意力
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = torch.sigmoid(attention)
        
        # 应用注意力
        x = x * attention
        return x 
