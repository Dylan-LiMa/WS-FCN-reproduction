import os
import requests
import zipfile
import tarfile
from tqdm import tqdm
import argparse
import shutil

def download_file(url, filename):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def extract_zip(zip_path, extract_path):
    """解压ZIP文件"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def extract_tar(tar_path, extract_path):
    """解压TAR文件"""
    with tarfile.open(tar_path, 'r') as tar_ref:
        tar_ref.extractall(extract_path)

def extract_tgz(tgz_path, extract_path):
    """解压TGZ文件"""
    with tarfile.open(tgz_path, 'r:gz') as tar_ref:
        tar_ref.extractall(extract_path)

def download_voc2012():
    """下载VOC2012数据集"""
    print('开始下载VOC2012数据集...')
    
    # 创建目录
    os.makedirs('data/voc', exist_ok=True)
    
    # 下载VOC2012数据集
    voc_url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    tar_path = 'data/voc/VOCtrainval_11-May-2012.tar'
    
    if not os.path.exists(tar_path):
        print('下载VOC2012数据集...')
        download_file(voc_url, tar_path)
    
    # 解压文件
    print('解压VOC2012数据集...')
    extract_tar(tar_path, 'data/voc')
    
    # 删除tar文件
    if os.path.exists(tar_path):
        os.remove(tar_path)
    
    print('VOC2012数据集下载和解压完成！')

def download_sbd():
    """下载SBD数据集"""
    print('开始下载SBD数据集...')
    
    # 创建目录
    os.makedirs('data/sbd', exist_ok=True)
    
    # 下载SBD数据集
    sbd_url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'
    tgz_path = 'data/sbd/benchmark.tgz'
    
    if not os.path.exists(tgz_path):
        print('下载SBD数据集...')
        download_file(sbd_url, tgz_path)
    
    # 解压文件
    print('解压SBD数据集...')
    extract_tgz(tgz_path, 'data/sbd')
    
    # 删除tgz文件
    if os.path.exists(tgz_path):
        os.remove(tgz_path)
    
    print('SBD数据集下载和解压完成！')

def download_pretrained_model():
    """下载预训练模型"""
    print('开始下载预训练模型...')
    
    # 创建目录
    os.makedirs('models/weights', exist_ok=True)
    
    # 下载预训练模型
    model_url = 'https://drive.google.com/uc?export=download&id=1qXqHXKX8qXqXqXqXqXqXqXqXqXqXqXqX'
    model_path = 'models/weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.pth'
    
    if not os.path.exists(model_path):
        print('下载预训练模型...')
        download_file(model_url, model_path)
    
    print('预训练模型下载完成！')

def main():
    parser = argparse.ArgumentParser(description='下载数据集和预训练模型')
    parser.add_argument('--dataset', type=str, choices=['voc2012', 'sbd', 'all'],
                      default='all', help='要下载的数据集名称')
    parser.add_argument('--pretrained', action='store_true',
                      help='是否下载预训练模型')
    args = parser.parse_args()
    
    if args.dataset in ['voc2012', 'all']:
        download_voc2012()
    
    if args.dataset in ['sbd', 'all']:
        download_sbd()
    
    if args.pretrained:
        download_pretrained_model()
    
    print('所有下载任务完成！')

if __name__ == '__main__':
    main() 