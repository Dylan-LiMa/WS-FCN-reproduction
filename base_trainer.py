import torch
from utils.checkpoints import CheckpointManager
from utils.metrics import Metrics
from utils.timer import Timer
from utils.stat_manager import StatManager

class BaseTrainer:
    def __init__(self, model, optimizer, device, config):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        self.checkpoint_manager = CheckpointManager(config['checkpoint_dir'])
        self.metrics = Metrics()
        self.timer = Timer()
        self.stat_manager = StatManager()
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        self.metrics.reset()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            self.metrics.update(loss.item(), output, target)
            
            if batch_idx % self.config['log_interval'] == 0:
                self.log_progress(batch_idx, len(train_loader))
    
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                self.metrics.update(loss.item(), output, target)
        
        return self.metrics.compute()
    
    def save_checkpoint(self, epoch):
        """保存检查点"""
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        self.checkpoint_manager.save_checkpoint(state, f'checkpoint_epoch_{epoch}.pth')
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        state = self.checkpoint_manager.load_checkpoint(filename)
        self.model.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        return state['epoch']
    
    def log_progress(self, batch_idx, total_batches):
        """记录训练进度"""
        metrics = self.metrics.compute()
        print(f'Train Batch: [{batch_idx}/{total_batches}] '
              f'Loss: {metrics["loss"]:.4f} '
              f'Precision: {metrics["precision"]:.4f} '
              f'Recall: {metrics["recall"]:.4f} '
              f'F1: {metrics["f1"]:.4f}') 