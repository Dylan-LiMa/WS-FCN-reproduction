import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import logging

class Inferencer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # 设置图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # 设置日志记录
        self._setup_logging()
        
        # 创建结果目录
        os.makedirs(config.result_dir, exist_ok=True)
    
    def _setup_logging(self):
        """设置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.log_dir, 'inference.log')),
                logging.StreamHandler()
            ]
        )
    
    def preprocess_image(self, image):
        """预处理图像"""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # 保存原始图像大小
        self.original_size = image.size
        
        # 应用变换
        image = self.transform(image)
        return image.unsqueeze(0)
    
    def postprocess_prediction(self, pred):
        """后处理预测结果"""
        # 调整大小到原始图像尺寸
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(0),
            size=self.original_size[::-1],
            mode='bilinear',
            align_corners=True
        ).squeeze(0)
        
        # 获取类别预测
        pred = pred.argmax(dim=0)
        return pred
    
    def predict(self, image):
        """预测单张图像"""
        # 预处理图像
        image = self.preprocess_image(image)
        image = image.to(self.device)
        
        # 模型推理
        with torch.no_grad():
            output = self.model(image)
        
        # 后处理预测结果
        pred = self.postprocess_prediction(output)
        return pred
    
    def visualize_prediction(self, image, pred, save_path=None):
        """可视化预测结果"""
        # 创建图像
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 原始图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        axes[0].imshow(image)
        axes[0].set_title('原始图像')
        axes[0].axis('off')
        
        # 预测结果
        pred_img = self.config.colormap[pred.cpu().numpy()]
        axes[1].imshow(pred_img)
        axes[1].set_title('预测结果')
        axes[1].axis('off')
        
        # 保存或显示图像
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def predict_batch(self, images):
        """批量预测图像"""
        results = []
        for image in images:
            pred = self.predict(image)
            results.append(pred)
        return results
    
    def predict_directory(self, image_dir, output_dir=None):
        """预测目录中的所有图像"""
        if output_dir is None:
            output_dir = os.path.join(self.config.result_dir, 'predictions')
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图像文件
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for image_file in tqdm(image_files, desc='预测图像'):
            # 读取图像
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path).convert('RGB')
            
            # 预测
            pred = self.predict(image)
            
            # 保存预测结果
            save_path = os.path.join(output_dir, f'pred_{image_file}')
            self.visualize_prediction(image, pred, save_path)
            
            # 保存预测掩码
            mask_path = os.path.join(output_dir, f'mask_{image_file}')
            pred_mask = Image.fromarray(pred.cpu().numpy().astype(np.uint8))
            pred_mask.save(mask_path) 