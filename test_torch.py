import torch
import torchvision

def test_torch():
    # 测试PyTorch版本
    print(f'PyTorch版本: {torch.__version__}')
    print(f'Torchvision版本: {torchvision.__version__}')
    
    # 测试CUDA是否可用
    print(f'CUDA是否可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA版本: {torch.version.cuda}')
        print(f'当前设备: {torch.cuda.get_device_name(0)}')
    
    # 测试张量操作
    x = torch.rand(5, 3)
    print('\n测试张量操作:')
    print(f'随机张量:\n{x}')
    print(f'张量形状: {x.shape}')
    print(f'张量设备: {x.device}')

if __name__ == '__main__':
    test_torch() 