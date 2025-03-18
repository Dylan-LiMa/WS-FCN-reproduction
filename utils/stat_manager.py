import numpy as np
from collections import defaultdict

class StatManager:
    def __init__(self):
        self.stats = defaultdict(list)
    
    def update(self, name, value):
        """更新统计值"""
        self.stats[name].append(value)
    
    def get_mean(self, name):
        """获取平均值"""
        values = self.stats[name]
        return np.mean(values) if values else 0
    
    def get_std(self, name):
        """获取标准差"""
        values = self.stats[name]
        return np.std(values) if values else 0
    
    def reset(self, name=None):
        """重置统计值"""
        if name is None:
            self.stats.clear()
        else:
            self.stats[name] = []
    
    def get_summary(self):
        """获取所有统计值的摘要"""
        summary = {}
        for name, values in self.stats.items():
            summary[name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        return summary 