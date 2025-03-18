import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_metrics(pred, target, num_classes):
    """计算语义分割的评估指标
    
    Args:
        pred: 预测的分割图 [B, H, W]
        target: 真实的分割图 [B, H, W]
        num_classes: 类别数量
    
    Returns:
        dict: 包含各项指标的字典
    """
    # 将预测和真实标签展平
    pred = pred.cpu().numpy().flatten()
    target = target.cpu().numpy().flatten()
    
    # 计算混淆矩阵
    cm = confusion_matrix(target, pred, labels=range(num_classes))
    
    # 计算每个类别的IoU
    ious = []
    for i in range(num_classes):
        intersection = cm[i, i]
        union = np.sum(cm[i, :]) + np.sum(cm[:, i]) - intersection
        if union == 0:
            ious.append(0)
        else:
            ious.append(intersection / union)
    
    # 计算平均IoU
    miou = np.mean(ious)
    
    # 计算像素准确率
    pixel_acc = np.sum(np.diag(cm)) / np.sum(cm)
    
    # 计算平均类别准确率
    class_acc = np.mean(np.diag(cm) / np.sum(cm, axis=1))
    
    return {
        'miou': miou,
        'pixel_acc': pixel_acc,
        'mean_acc': class_acc,
        'ious': ious
    }

def calculate_boundary_metrics(pred, target, num_classes):
    """计算边界相关的评估指标
    
    Args:
        pred: 预测的分割图 [B, H, W]
        target: 真实的分割图 [B, H, W]
        num_classes: 类别数量
    
    Returns:
        dict: 包含边界指标的字典
    """
    # 计算边界准确率
    pred_boundary = get_boundary(pred)
    target_boundary = get_boundary(target)
    
    # 计算边界IoU
    boundary_intersection = np.sum(pred_boundary * target_boundary)
    boundary_union = np.sum(pred_boundary) + np.sum(target_boundary) - boundary_intersection
    boundary_iou = boundary_intersection / boundary_union if boundary_union > 0 else 0
    
    # 计算边界准确率
    boundary_acc = np.sum(pred_boundary == target_boundary) / pred_boundary.size
    
    return {
        'boundary_iou': boundary_iou,
        'boundary_acc': boundary_acc
    }

def get_boundary(mask):
    """获取分割图的边界
    
    Args:
        mask: 分割图 [B, H, W]
    
    Returns:
        boundary: 边界图 [B, H, W]
    """
    # 使用Sobel算子计算梯度
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    
    # 将mask转换为tensor
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    
    # 计算x和y方向的梯度
    grad_x = torch.nn.functional.conv2d(
        mask.unsqueeze(0).float(),
        sobel_x.view(1, 1, 3, 3),
        padding=1
    )
    grad_y = torch.nn.functional.conv2d(
        mask.unsqueeze(0).float(),
        sobel_y.view(1, 1, 3, 3),
        padding=1
    )
    
    # 计算梯度幅值
    grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    
    # 二值化得到边界
    boundary = (grad > 0).squeeze(0).numpy()
    
    return boundary

def calculate_miou(pred, target, num_classes):
    """计算平均交并比(mIoU)
    
    Args:
        pred: 预测结果 [B, H, W] 或 [H, W]
        target: 真实标签 [B, H, W] 或 [H, W]
        num_classes: 类别数量
        
    Returns:
        float: mIoU值
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
        
    # 确保输入是2D数组
    if len(pred.shape) == 3:
        pred = pred.reshape(-1)
        target = target.reshape(-1)
    
    # 计算混淆矩阵
    cm = confusion_matrix(target, pred, labels=range(num_classes))
    
    # 计算每个类别的IoU
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - intersection
    iou = intersection / (union + 1e-10)
    
    # 忽略背景类(0)
    iou = iou[1:]
    
    return np.mean(iou)

def calculate_pixacc(pred, target):
    """计算像素精度(Pixel Accuracy)
    
    Args:
        pred: 预测结果 [B, H, W] 或 [H, W]
        target: 真实标签 [B, H, W] 或 [H, W]
        
    Returns:
        float: 像素精度
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
        
    # 确保输入是2D数组
    if len(pred.shape) == 3:
        pred = pred.reshape(-1)
        target = target.reshape(-1)
    
    correct = (pred == target).sum()
    total = pred.size
    
    return correct / total

def count_parameters(model):
    """计算模型参数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        int: 参数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 