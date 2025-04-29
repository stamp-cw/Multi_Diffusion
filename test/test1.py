import os
from torchvision import datasets
import torchvision.transforms as transforms
import torch

# 定义数据转换
transform = transforms.Compose([
    transforms.CenterCrop(178),  # 先裁剪成正方形
    transforms.Resize(128),  # 缩放到128×128
    transforms.ToTensor(),  # 必须启用，将PIL图像转为Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 加载CelebA数据集
dataset = datasets.CelebA(
    root=os.path.normpath(r"D:\Project\Multi_Diffusion\data"),
    split='train',
    target_type='attr',
    transform=transform,  # 必须启用transform
    download=True
)

# 创建DataLoader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# 测试迭代
for images, labels in dataloader:
    print(images.shape)  # 应输出 torch.Size([32, 3, 218, 178])
    print(labels.shape)  # 应输出 torch.Size([32, 40])
    break