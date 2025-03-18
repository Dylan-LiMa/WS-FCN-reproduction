import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255):
        super(SegmentationLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # 交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction='mean'
        )
        
        # Dice损失
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index
        )
        
        # 边界感知损失
        self.boundary_loss = BoundaryLoss(
            num_classes=num_classes,
            ignore_index=ignore_index
        )
    
    def forward(self, pred, target):
        # 计算交叉熵损失
        ce_loss = self.ce_loss(pred, target)
        
        # 计算Dice损失
        dice_loss = self.dice_loss(pred, target)
        
        # 计算边界感知损失
        boundary_loss = self.boundary_loss(pred, target)
        
        # 总损失
        total_loss = ce_loss + 0.5 * dice_loss + 0.5 * boundary_loss
        
        return total_loss

class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        # 将预测转换为one-hot格式
        pred = F.softmax(pred, dim=1)
        
        # 创建mask
        mask = (target != self.ignore_index)
        
        # 计算每个类别的Dice损失
        dice_loss = 0
        for i in range(self.num_classes):
            pred_i = pred[:, i]
            target_i = (target == i).float()
            
            # 应用mask
            pred_i = pred_i[mask]
            target_i = target_i[mask]
            
            # 计算Dice系数
            intersection = (pred_i * target_i).sum()
            dice = (2. * intersection + 1e-6) / (pred_i.sum() + target_i.sum() + 1e-6)
            
            dice_loss += 1 - dice
        
        return dice_loss / self.num_classes

class BoundaryLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255):
        super(BoundaryLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        # 将预测转换为one-hot格式
        pred = F.softmax(pred, dim=1)
        
        # 创建mask
        mask = (target != self.ignore_index)
        
        # 计算边界损失
        boundary_loss = 0
        for i in range(self.num_classes):
            pred_i = pred[:, i]
            target_i = (target == i).float()
            
            # 计算边界
            pred_boundary = self._get_boundary(pred_i)
            target_boundary = self._get_boundary(target_i)
            
            # 应用mask
            pred_boundary = pred_boundary[mask]
            target_boundary = target_boundary[mask]
            
            # 计算边界损失
            boundary_loss += F.binary_cross_entropy(pred_boundary, target_boundary)
        
        return boundary_loss / self.num_classes
    
    def _get_boundary(self, x):
        """计算边界"""
        # 使用Sobel算子计算梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        if x.is_cuda:
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()
        
        # 计算x和y方向的梯度
        grad_x = F.conv2d(x.unsqueeze(0), sobel_x, padding=1)
        grad_y = F.conv2d(x.unsqueeze(0), sobel_y, padding=1)
        
        # 计算梯度幅值
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # 归一化
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-6)
        
        return grad.squeeze(0) 
