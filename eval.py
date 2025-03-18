import os
import torch
import argparse
from core.config import Config
from core.evaluator import Evaluator
from datasets import get_dataset
from models import SegmentationModel

def parse_args():
    parser = argparse.ArgumentParser(description='评估语义分割模型')
    parser.add_argument('--dataset', type=str, default='voc', choices=['voc', 'sbd'],
                      help='选择数据集 (voc 或 sbd)')
    parser.add_argument('--data-root', type=str, default='data',
                      help='数据集根目录')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='模型检查点路径')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='批次大小')
    parser.add_argument('--num-samples', type=int, default=5,
                      help='可视化样本数量')
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建配置
    config = Config(
        dataset_name=args.dataset,
        dataset_root=args.data_root,
        batch_size=args.batch_size
    )
    
    # 创建数据集
    dataset = get_dataset(
        name=config.dataset_name,
        root_dir=config.dataset_root,
        split='val'
    )
    
    # 创建模型
    model = SegmentationModel(
        backbone=config.backbone,
        num_classes=config.num_classes,
        pretrained=False
    )
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'加载检查点: {args.checkpoint}')
    
    # 创建评估器
    evaluator = Evaluator(model, dataset, config)
    
    # 评估模型
    metrics = evaluator.evaluate()
    
    # 可视化结果
    evaluator.visualize_results(args.num_samples)
    
    print('\n评估完成！')
    print(f'平均损失: {metrics["loss"]:.4f}')
    print(f'平均IoU: {metrics["mIoU"]:.4f}')
    print(f'平均准确率: {metrics["mAcc"]:.4f}')

if __name__ == '__main__':
    main() 