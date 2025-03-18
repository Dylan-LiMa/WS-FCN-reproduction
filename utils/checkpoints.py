import os
import torch

class CheckpointManager:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_checkpoint(self, state, filename):
        """保存检查点"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save(state, filepath)
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        filepath = os.path.join(self.save_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"检查点文件不存在: {filepath}")
        return torch.load(filepath)
    
    def get_latest_checkpoint(self):
        """获取最新的检查点文件"""
        checkpoints = [f for f in os.listdir(self.save_dir) if f.endswith('.pth')]
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda x: os.path.getctime(os.path.join(self.save_dir, x))) 