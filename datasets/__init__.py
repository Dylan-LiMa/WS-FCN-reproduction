from .base_dataset import BaseDataset
from .voc import VOC2012Dataset
from .sbd import SBDDataset
from .transforms import get_train_transforms, get_val_transforms

def get_dataset(name, root_dir, split='train', transform=None):
    """
    获取数据集实例
    
    Args:
        name: 数据集名称 ('voc' 或 'sbd')
        root_dir: 数据集根目录
        split: 数据集分割 ('train' 或 'val')
        transform: 数据转换函数
    
    Returns:
        Dataset: 数据集实例
    """
    if transform is None:
        transform = get_train_transforms() if split == 'train' else get_val_transforms()
    
    if name.lower() == 'voc':
        return VOC2012Dataset(root_dir, split, transform)
    elif name.lower() == 'sbd':
        return SBDDataset(root_dir, split, transform)
    else:
        raise ValueError(f'不支持的数据集: {name}') 