import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from models import WSFCN
from datasets.voc_dataset import VOCDataset
from utils.metrics import calculate_miou, calculate_pixacc, count_parameters
from config import Config

def train(config):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建数据集和数据加载器
    train_dataset = VOCDataset(config.data_root, split='train', transform=config.train_transform)
    val_dataset = VOCDataset(config.data_root, split='val', transform=config.val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                            shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                          shuffle=False, num_workers=config.num_workers)
    
    # 创建模型
    model = WSFCN(num_classes=config.num_classes).to(device)
    
    # 计算参数量
    params = count_parameters(model)
    print(f'模型参数量: {params:,}')
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(config.log_dir)
    
    # 训练循环
    best_miou = 0
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_miou = 0
        train_pixacc = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.num_epochs}')
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            # 计算指标
            preds = outputs.argmax(dim=1)
            miou = calculate_miou(preds, targets, config.num_classes)
            pixacc = calculate_pixacc(preds, targets)
            
            train_loss += loss.item()
            train_miou += miou
            train_pixacc += pixacc
            
            pbar.set_postfix({'loss': loss.item(), 'mIoU': miou, 'PixAcc': pixacc})
        
        # 计算平均指标
        train_loss /= len(train_loader)
        train_miou /= len(train_loader)
        train_pixacc /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_miou = 0
        val_pixacc = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                preds = outputs.argmax(dim=1)
                miou = calculate_miou(preds, targets, config.num_classes)
                pixacc = calculate_pixacc(preds, targets)
                
                val_loss += loss.item()
                val_miou += miou
                val_pixacc += pixacc
        
        val_loss /= len(val_loader)
        val_miou /= len(val_loader)
        val_pixacc /= len(val_loader)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('mIoU/train', train_miou, epoch)
        writer.add_scalar('mIoU/val', val_miou, epoch)
        writer.add_scalar('PixAcc/train', train_pixacc, epoch)
        writer.add_scalar('PixAcc/val', val_pixacc, epoch)
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{config.num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}, Train PixAcc: {train_pixacc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}, Val PixAcc: {val_pixacc:.4f}')
        
        # 保存最佳模型
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'params': params
            }, os.path.join(config.checkpoint_dir, 'best_model.pth'))
    
    writer.close()

if __name__ == '__main__':
    config = Config()
    train(config) 