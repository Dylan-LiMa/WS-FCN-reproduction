import argparse
import yaml
import os

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练和评估模型')
    
    # 基本参数
    parser.add_argument('--config', type=str, required=True,
                      help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'],
                      default='train', help='运行模式')
    
    # 训练相关参数
    parser.add_argument('--resume', type=str,
                      help='恢复训练的检查点路径')
    parser.add_argument('--epochs', type=int,
                      help='训练轮数')
    parser.add_argument('--batch_size', type=int,
                      help='批次大小')
    parser.add_argument('--learning_rate', type=float,
                      help='学习率')
    
    # 评估相关参数
    parser.add_argument('--checkpoint', type=str,
                      help='评估用的模型检查点路径')
    parser.add_argument('--output', type=str,
                      help='评估结果输出路径')
    
    # 数据集相关参数
    parser.add_argument('--dataset', type=str, choices=['coco', 'voc'],
                      help='数据集名称')
    parser.add_argument('--train_data_dir', type=str,
                      help='训练数据目录')
    parser.add_argument('--val_data_dir', type=str,
                      help='验证数据目录')
    
    args = parser.parse_args()
    
    # 加载配置文件
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新配置
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    
    return config

if __name__ == '__main__':
    config = parse_args()
    print('配置信息:')
    print(yaml.dump(config, default_flow_style=False)) 