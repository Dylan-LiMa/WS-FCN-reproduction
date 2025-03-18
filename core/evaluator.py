import os
import torch
import numpy as np
from tqdm import tqdm
import logging
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class Evaluator:
    def __init__(self, model, dataset, config):
        self.model = model
        self.dataset = dataset
        self.config = config
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # 创建数据加载器
        self.data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
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
                logging.FileHandler(os.path.join(self.config.log_dir, 'eval.log')),
                logging.StreamHandler()
            ]
        )
    
    def evaluate(self):
        """评估模型性能"""
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(self.data_loader, desc='Evaluating'):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                loss = self.model.loss_fn(outputs, targets)
                
                # 收集预测结果
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                total_loss += loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(self.data_loader)
        
        # 计算评估指标
        metrics = self._compute_metrics(np.array(all_preds), np.array(all_targets))
        metrics['loss'] = avg_loss
        
        # 记录评估结果
        self._log_metrics(metrics)
        
        # 保存混淆矩阵
        self._save_confusion_matrix(np.array(all_preds), np.array(all_targets))
        
        return metrics
    
    def _compute_metrics(self, preds, targets):
        """计算评估指标"""
        metrics = {}
        
        # 计算每个类别的IoU
        ious = []
        for i in range(self.config.num_classes):
            pred_mask = (preds == i)
            target_mask = (targets == i)
            intersection = np.sum(pred_mask & target_mask)
            union = np.sum(pred_mask | target_mask)
            iou = intersection / (union + 1e-10)
            ious.append(iou)
        
        # 计算平均IoU
        metrics['mIoU'] = np.mean(ious)
        
        # 计算每个类别的准确率
        accuracies = []
        for i in range(self.config.num_classes):
            pred_mask = (preds == i)
            target_mask = (targets == i)
            accuracy = np.sum(pred_mask == target_mask) / np.sum(target_mask)
            accuracies.append(accuracy)
        
        # 计算平均准确率
        metrics['mAcc'] = np.mean(accuracies)
        
        return metrics
    
    def _log_metrics(self, metrics):
        """记录评估指标"""
        logging.info('评估结果:')
        logging.info(f'平均损失: {metrics["loss"]:.4f}')
        logging.info(f'平均IoU: {metrics["mIoU"]:.4f}')
        logging.info(f'平均准确率: {metrics["mAcc"]:.4f}')
        
        # 记录每个类别的IoU
        for i, iou in enumerate(metrics.get('ious', [])):
            logging.info(f'类别 {i} ({self.dataset.class_names[i]}) IoU: {iou:.4f}')
    
    def _save_confusion_matrix(self, preds, targets):
        """保存混淆矩阵"""
        cm = confusion_matrix(targets.flatten(), preds.flatten())
        plt.figure(figsize=(15, 15))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.savefig(os.path.join(self.config.result_dir, 'confusion_matrix.png'))
        plt.close()
    
    def visualize_results(self, num_samples=5):
        """可视化预测结果"""
        self.model.eval()
        with torch.no_grad():
            for i, (images, targets) in enumerate(self.data_loader):
                if i >= num_samples:
                    break
                
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # 获取预测结果
                outputs = self.model(images)
                preds = outputs.argmax(dim=1)
                
                # 可视化每个样本
                for j in range(images.size(0)):
                    # 创建图像
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # 原始图像
                    img = images[j].cpu().permute(1, 2, 0)
                    img = (img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406]))
                    axes[0].imshow(img)
                    axes[0].set_title('原始图像')
                    axes[0].axis('off')
                    
                    # 真实标签
                    target = targets[j].cpu()
                    target_img = self.dataset.colormap[target]
                    axes[1].imshow(target_img)
                    axes[1].set_title('真实标签')
                    axes[1].axis('off')
                    
                    # 预测结果
                    pred = preds[j].cpu()
                    pred_img = self.dataset.colormap[pred]
                    axes[2].imshow(pred_img)
                    axes[2].set_title('预测结果')
                    axes[2].axis('off')
                    
                    # 保存图像
                    plt.savefig(os.path.join(self.config.result_dir, f'result_{i}_{j}.png'))
                    plt.close() 