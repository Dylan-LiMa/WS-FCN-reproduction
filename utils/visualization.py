import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

def visualize_prediction(rgb, depth, target, pred, save_dir):
    """可视化预测结果
    
    Args:
        rgb: RGB图像 [C, H, W]
        depth: 深度图像 [1, H, W]
        target: 真实标签 [H, W]
        pred: 预测标签 [H, W]
        save_dir: 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 将tensor转换为numpy数组
    rgb = rgb.numpy().transpose(1, 2, 0)
    depth = depth.squeeze().numpy()
    target = target.numpy()
    pred = pred.numpy()
    
    # 反归一化RGB图像
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb = rgb * std + mean
    rgb = np.clip(rgb, 0, 1)
    
    # 创建图像网格
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 显示RGB图像
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Image')
    axes[0, 0].axis('off')
    
    # 显示深度图像
    axes[0, 1].imshow(depth, cmap='gray')
    axes[0, 1].set_title('Depth Image')
    axes[0, 1].axis('off')
    
    # 显示真实标签
    axes[1, 0].imshow(target, cmap='tab20')
    axes[1, 0].set_title('Ground Truth')
    axes[1, 0].axis('off')
    
    # 显示预测结果
    axes[1, 1].imshow(pred, cmap='tab20')
    axes[1, 1].set_title('Prediction')
    axes[1, 1].axis('off')
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'prediction.png'))
    plt.close()

def visualize_attention_maps(model, rgb, depth, save_dir):
    """可视化注意力图
    
    Args:
        model: 模型
        rgb: RGB图像 [C, H, W]
        depth: 深度图像 [1, H, W]
        save_dir: 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 将输入移到设备
    rgb = rgb.unsqueeze(0).to(model.device)
    depth = depth.unsqueeze(0).to(model.device)
    
    # 获取注意力图
    with torch.no_grad():
        # 提取RGB特征
        rgb_features = model.rgb_encoder(rgb)
        
        # 提取深度特征
        depth_features = model.depth_encoder(depth)
        
        # 获取注意力图
        spatial_attention = model.fusion.attention.spatial_attention(rgb_features[-1])
        channel_attention = model.fusion.attention.channel_attention(rgb_features[-1])
    
    # 将注意力图转换为numpy数组
    spatial_attention = spatial_attention.squeeze().cpu().numpy()
    channel_attention = channel_attention.squeeze().cpu().numpy()
    
    # 创建图像网格
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 显示空间注意力图
    axes[0].imshow(spatial_attention, cmap='jet')
    axes[0].set_title('Spatial Attention')
    axes[0].axis('off')
    
    # 显示通道注意力图
    axes[1].imshow(channel_attention, cmap='jet')
    axes[1].set_title('Channel Attention')
    axes[1].axis('off')
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'attention_maps.png'))
    plt.close()

def visualize_feature_maps(model, rgb, depth, save_dir):
    """可视化特征图
    
    Args:
        model: 模型
        rgb: RGB图像 [C, H, W]
        depth: 深度图像 [1, H, W]
        save_dir: 保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 将输入移到设备
    rgb = rgb.unsqueeze(0).to(model.device)
    depth = depth.unsqueeze(0).to(model.device)
    
    # 获取特征图
    with torch.no_grad():
        # 提取RGB特征
        rgb_features = model.rgb_encoder(rgb)
        
        # 提取深度特征
        depth_features = model.depth_encoder(depth)
        
        # 特征融合
        fused_features = model.fusion(rgb_features[-1], depth_features[-1])
    
    # 将特征图转换为numpy数组
    rgb_features = rgb_features[-1].squeeze().cpu().numpy()
    depth_features = depth_features[-1].squeeze().cpu().numpy()
    fused_features = fused_features.squeeze().cpu().numpy()
    
    # 创建图像网格
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 显示RGB特征图
    axes[0].imshow(rgb_features[0], cmap='jet')
    axes[0].set_title('RGB Features')
    axes[0].axis('off')
    
    # 显示深度特征图
    axes[1].imshow(depth_features[0], cmap='jet')
    axes[1].set_title('Depth Features')
    axes[1].axis('off')
    
    # 显示融合特征图
    axes[2].imshow(fused_features[0], cmap='jet')
    axes[2].set_title('Fused Features')
    axes[2].axis('off')
    
    # 保存图像
    plt.savefig(os.path.join(save_dir, 'feature_maps.png'))
    plt.close() 