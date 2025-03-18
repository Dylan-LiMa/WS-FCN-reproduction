import torch
from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    """创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def collate_fn(batch):
    """自定义批处理函数"""
    # 这里需要根据具体任务实现批处理逻辑
    return batch 