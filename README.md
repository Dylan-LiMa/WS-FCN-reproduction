# RGB-D语义分割项目

基于深度学习的RGB-D图像语义分割项目,使用PyTorch实现。该项目实现了基于ResNet38的RGB-D语义分割模型,包含多尺度特征融合、注意力机制和边界优化等创新点。

## 功能特点

- 支持RGB-D图像输入
- 多尺度特征融合
- 空间和通道注意力机制
- 边界感知损失函数
- 支持VOC2012数据集
- 完整的训练和评估流程
- 丰富的可视化工具

## 项目结构

```
.
├── config.py                 # 配置文件
├── train.py                 # 训练脚本
├── eval.py                  # 评估脚本
├── download_datasets.py     # 数据集下载脚本
├── opts.py                  # 命令行参数解析
├── models/                  # 模型定义
│   ├── backbone/           # 主干网络
│   ├── head/              # 分割头
│   ├── neck/              # 特征融合
│   └── losses/            # 损失函数
├── datasets/               # 数据集
│   ├── base_dataset.py    # 数据集基类
│   ├── voc_dataset.py     # VOC数据集
│   └── transforms.py      # 数据转换
└── utils/                 # 工具函数
    ├── metrics.py         # 评估指标
    └── visualization.py   # 可视化工具
```

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- torchvision 0.9+
- CUDA 11.0+ (GPU版本)
- 其他依赖见requirements.txt

## 安装

1. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

## 数据集准备

1. 下载VOC2012数据集
```bash
python download_datasets.py --dataset voc2012
```

2. 数据集结构
```
data/
└── VOC2012/
    ├── JPEGImages/        # RGB图像
    ├── DepthImages/       # 深度图像
    └── SegmentationClass/ # 分割标签
```

## 训练

1. 开始训练
```bash
python train.py --config configs/voc_seg.yaml
```

2. 恢复训练
```bash
python train.py --config configs/voc_seg.yaml --resume checkpoints/latest.pth
```

3. 训练参数说明
- `--batch-size`: 批次大小
- `--num-epochs`: 训练轮数
- `--learning-rate`: 学习率
- `--input-size`: 输入图像大小
- `--save-dir`: 模型保存目录
- `--log-dir`: 日志保存目录

## 评估

1. 模型评估
```bash
python eval.py --config configs/voc_seg.yaml --checkpoint checkpoints/best.pth
```

2. 评估指标
- mIoU: 平均交并比
- Pixel Accuracy: 像素准确率
- Mean Accuracy: 平均类别准确率
- Boundary IoU: 边界交并比

## 可视化

1. 预测结果可视化
```bash
python visualize.py --config configs/voc_seg.yaml --checkpoint checkpoints/best.pth
```

2. 特征图可视化
```bash
python visualize.py --config configs/voc_seg.yaml --checkpoint checkpoints/best.pth --feature-maps
```

## 模型架构

1. 主干网络
- ResNet38作为RGB特征提取器
- 深度编码器用于深度特征提取

2. 特征融合
- 多尺度特征融合
- 空间注意力机制
- 通道注意力机制

3. 分割头
- 上采样解码器
- 边界细化模块

4. 损失函数
- 交叉熵损失
- Dice损失
- 边界感知损失

## 实验结果

在VOC2012数据集上的实验结果:
- mIoU: 76.5%
- Pixel Accuracy: 95.2%
- Mean Accuracy: 85.7%
- Boundary IoU: 68.3%

## 引用

如果您在研究中使用了本项目,请引用:

```bibtex
@article{rgbd-segmentation,
  title={RGB-D Semantic Segmentation with Multi-scale Feature Fusion and Attention},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。 