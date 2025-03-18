import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from models import RGBDSegmentationModel
from datasets.transforms import get_val_transforms
from utils.visualization import visualize_prediction
from config import Config
import argparse
import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='使用训练好的模型进行语义分割推理')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True,
                      help='输入图像路径或目录')
    parser.add_argument('--output', type=str, default=None,
                      help='输出目录路径')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='批次大小')
    return parser.parse_args()

class Predictor:
    def __init__(self, config, checkpoint_path):
        """初始化预测器
        
        Args:
            config: 配置对象
            checkpoint_path: 模型检查点路径
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载模型
        self.model = RGBDSegmentationModel(
            num_classes=config.num_classes,
            pretrained=False
        ).to(self.device)
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 设置数据转换
        self.transform = get_val_transforms(config.input_size)
    
    def predict_single(self, rgb_path, depth_path, save_path=None):
        """预测单张图片
        
        Args:
            rgb_path: RGB图像路径
            depth_path: 深度图像路径
            save_path: 结果保存路径
            
        Returns:
            tuple: (预测结果, 可视化图像)
        """
        # 加载图像
        rgb = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path).convert('L')
        
        # 应用转换
        rgb, depth, _ = self.transform(rgb, depth, None)
        
        # 添加批次维度
        rgb = rgb.unsqueeze(0).to(self.device)
        depth = depth.unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(rgb, depth)
            pred = output.argmax(dim=1)
        
        # 转换为numpy数组
        pred = pred.squeeze().cpu().numpy()
        
        # 可视化
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            visualize_prediction(
                rgb.squeeze().cpu(),
                depth.squeeze().cpu(),
                None,  # 没有真实标签
                pred,
                save_path
            )
        
        return pred
    
    def predict_batch(self, rgb_dir, depth_dir, save_dir):
        """预测整个文件夹的图片
        
        Args:
            rgb_dir: RGB图像目录
            depth_dir: 深度图像目录
            save_dir: 结果保存目录
        """
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取所有RGB图像
        rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith(('.jpg', '.png'))]
        
        for rgb_file in tqdm.tqdm(rgb_files, desc='Predicting'):
            # 构建文件路径
            rgb_path = os.path.join(rgb_dir, rgb_file)
            depth_path = os.path.join(depth_dir, rgb_file.replace('.jpg', '.png'))
            save_path = os.path.join(save_dir, rgb_file.replace('.jpg', '_pred.png'))
            
            # 预测
            self.predict_single(rgb_path, depth_path, save_path)

def main():
    # 解析命令行参数
    config = parse_args()
    
    # 创建预测器
    predictor = Predictor(config, config.checkpoint)
    
    if config.image_path and config.depth_path:
        # 单张图片预测
        predictor.predict_single(
            config.image_path,
            config.depth_path,
            config.save_path
        )
    elif config.image_dir and config.depth_dir:
        # 批量预测
        predictor.predict_batch(
            config.image_dir,
            config.depth_dir,
            config.save_dir
        )
    else:
        print('请指定输入图片路径或目录')

if __name__ == '__main__':
    main() 