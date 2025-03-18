import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import logging

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        # 设置优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 设置学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        # 设置损失函数
        self.criterion = model.loss_fn
        
        # 创建日志记录器
        self.writer = SummaryWriter(config.log_dir)
        self._setup_logging()
        
        # 初始化最佳模型保存
        self.best_val_loss = float('inf')
        self.best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
        
    def _setup_logging(self):
        """设置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.log_dir, 'train.log')),
                logging.StreamHandler()
            ]
        )
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 更新统计信息
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # 记录训练损失
            self.writer.add_scalar('Loss/train', loss.item(), self.current_epoch * len(self.train_loader) + batch_idx)
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                # 前向传播
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, val_loss):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.config.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(checkpoint, self.best_model_path)
            logging.info(f'保存最佳模型，验证损失: {val_loss:.4f}')
    
    def train(self, num_epochs):
        """训练模型"""
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            logging.info(f'Epoch {epoch+1}/{num_epochs}')
            
            # 训练一个epoch
            train_loss = self.train_epoch()
            logging.info(f'训练损失: {train_loss:.4f}')
            
            # 验证
            val_loss = self.validate()
            logging.info(f'验证损失: {val_loss:.4f}')
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 保存检查点
            self.save_checkpoint(epoch, val_loss)
            
            # 记录学习率
            self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        self.writer.close() 