import torch
from torch.utils.data import DataLoader
from datasets.coco import COCODataset
from core.config import Config
from utils.metrics import Metrics
import argparse
import json
import os

def evaluate(model, data_loader, device):
    """评估模型"""
    model.eval()
    metrics = Metrics()
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = model.criterion(output, target)
            metrics.update(loss.item(), output, target)
    
    return metrics.compute()

def main():
    parser = argparse.ArgumentParser(description='评估COCO分割模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output', type=str, help='评估结果输出路径')
    args = parser.parse_args()
    
    # 加载配置
    config = Config(args.config)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建数据集和数据加载器
    val_dataset = COCODataset(
        root_dir=config['val_data_dir'],
        ann_file=config['val_ann_file']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # 加载模型
    model = create_model(config)  # 需要实现
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 评估模型
    metrics = evaluate(model, val_loader, device)
    
    # 保存评估结果
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=4)
    
    # 打印评估结果
    print('\nEvaluation Results:')
    print(f'Loss: {metrics["loss"]:.4f}')
    print(f'Precision: {metrics["precision"]:.4f}')
    print(f'Recall: {metrics["recall"]:.4f}')
    print(f'F1 Score: {metrics["f1"]:.4f}')

if __name__ == '__main__':
    main() 