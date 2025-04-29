import json
import math
import os.path

import numpy
import torch
import torch_fidelity
from torch.utils.data import RandomSampler
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision import datasets


####################################################################################################
# 时间变化曲线函数
####################################################################################################

# beta schedule
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def binomial_linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def negative_binomial_linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.9
    beta_end = scale * 0.000996
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

def gaussian_linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def gamma_linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

def possion_linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.1
    beta_end = scale * 0.2
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

def optimize_gamma_linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sqrt_beta_schedule(timesteps, s=0.0001):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2205.14217
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = 1 - torch.sqrt(t + s)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


####################################################################################################
# config文件的导入导出函数
####################################################################################################
def export_config(dictionary, filename):
    """
    将字典保存为JSON文件。

    :param dictionary: 要保存的字典
    :param filename: 保存的文件名
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, ensure_ascii=False, indent=4)
        print(f"字典已成功保存为 {filename}")
    except Exception as e:
        print(f"保存文件时出错: {e}")

def import_config(filename):
    """
    将JSON文件导入为字典。

    :param filename: 要导入的JSON文件名
    :return: 导入的字典
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"JSON文件 {filename} 已成功导入为字典")
        return data
    except Exception as e:
        print(f"导入文件时出错: {e}")
        return None

####################################################################################################
# paint 绘图
####################################################################################################


import matplotlib.pyplot as plt
import numpy as np

def plot_images(imgs,img_num=8,time_steps=1000):
    # 画图
    fig = plt.figure(figsize=(img_num*2, img_num+2), constrained_layout=True)
    gs = fig.add_gridspec(img_num*2, img_num+2)
    t_idx = [x*(time_steps//img_num) for x in range(img_num)]
    t_idx[-1] = time_steps - 21
    t_idx.append(time_steps-11)
    t_idx.append(time_steps-2)
    for n_row in range(img_num):
        for n_col in range(img_num+2):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            img = torch.tensor(imgs[t_idx[n_col],n_row]).permute([1,2,0])
            # img = numpy.array((img - img.min()) / (img.max() - img.min()), dtype=numpy.uint8)
            img = numpy.array((img + 1.0) * 255 / 2, dtype=numpy.uint8)
            f_ax.imshow(img)
            f_ax.axis("off")
    for n_row in range(img_num,img_num*2):
        for n_col in range(img_num+2):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            img = torch.tensor(imgs[t_idx[n_col],n_row-img_num]).permute([1,2,0])
            img = numpy.array(((img - img.min()) / (img.max() - img.min()))*255, dtype=numpy.uint8)
            # img = numpy.array((img + 1.0) * 255 / 2, dtype=numpy.uint8)
            f_ax.imshow(img)
            f_ax.axis("off")
    return fig

####################################################################################################
# 画图
####################################################################################################

# 随机生成n*n张图
def generate_image(nb_diffusion,model,img_num=8,dataset_channel=1,dataset_image_size=32):
    generated_images = nb_diffusion.sample(model, 32, batch_size=img_num*img_num, channels=1)
    paint_images_1(generated_images,img_num=img_num)

# 画n*n张图
def paint_images_1(generated_images,img_num=8):
    # 画图
    fig = plt.figure(figsize=(img_num, img_num), constrained_layout=True)
    gs = fig.add_gridspec(img_num, img_num)
    imgs = generated_images[998].reshape(1, img_num*img_num, 1, 32, 32)
    for n_row in range(img_num):
        for n_col in range(img_num):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            img = imgs[0, n_row * img_num + n_col].transpose([1, 2, 0])
            img = numpy.array(((img - img.min()) / (img.max() - img.min()))*255, dtype=numpy.uint8)
            # print(img.min())
            # print(img.max())
            # img = numpy.array((img + 4.5*pow(10,3)) / pow(10,4) * 255, dtype=numpy.uint8)
            f_ax.imshow(img)
            f_ax.axis("off")
    plt.show()

def paint_images_2(imgs,img_num=8,time_steps=1000):
    # 画图
    fig = plt.figure(figsize=(img_num, img_num+2), constrained_layout=True)
    gs = fig.add_gridspec(img_num, img_num+2)
    t_idx = [x*(time_steps//img_num) for x in range(img_num)]
    t_idx[-1] = time_steps - 11
    t_idx.append(time_steps-2)
    t_idx.append(time_steps-1)
    for n_row in range(img_num):
        for n_col in range(img_num+2):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            img = torch.tensor(imgs[t_idx[n_col],n_row]).permute([1,2,0])
            # img = numpy.array((img - img.min()) / (img.max() - img.min()), dtype=numpy.uint8)
            img = numpy.array((img + 1.0) * 255 / 2, dtype=numpy.uint8)
            f_ax.imshow(img)
            f_ax.axis("off")
    plt.show()

def paint_images_3(imgs,img_num=8,time_steps=1000):
    # 画图
    fig = plt.figure(figsize=(img_num*2, img_num+2), constrained_layout=True)
    gs = fig.add_gridspec(img_num*2, img_num+2)
    t_idx = [x*(time_steps//img_num) for x in range(img_num)]
    t_idx[-1] = time_steps - 11
    t_idx.append(time_steps-2)
    t_idx.append(time_steps-1)
    for n_row in range(img_num):
        for n_col in range(img_num+2):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            img = torch.tensor(imgs[t_idx[n_col],n_row]).permute([1,2,0])
            # img = numpy.array((img - img.min()) / (img.max() - img.min()), dtype=numpy.uint8)
            img = numpy.array((img + 1.0) * 255 / 2, dtype=numpy.uint8)
            f_ax.imshow(img)
            f_ax.axis("off")
    for n_row in range(img_num,img_num*2):
        for n_col in range(img_num+2):
            f_ax = fig.add_subplot(gs[n_row, n_col])
            img = torch.tensor(imgs[t_idx[n_col],n_row-img_num]).permute([1,2,0])
            img = numpy.array(((img - img.min()) / (img.max() - img.min()))*255, dtype=numpy.uint8)
            # img = numpy.array((img + 1.0) * 255 / 2, dtype=numpy.uint8)
            f_ax.imshow(img)
            f_ax.axis("off")
    plt.show()

# def paint_images_4(imgs,img_num=8,time_steps=1000):
#
#     transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.Pad(padding=2),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5], std=[0.5]),
#     ])
#     root_dir=rf"D:\Project\Multi_Diffusion"
#     dataset = datasets.MNIST(rf"{root_dir}/data", train=True, download=True, transform=transform)
#     dataset_channel = 1
#     dataset_image_size = 32
#
#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
#     images, labels = next(iter(train_loader))
#     # 画图
#     fig = plt.figure(figsize=(img_num*2, img_num+2), constrained_layout=True)
#     gs = fig.add_gridspec(img_num*2, img_num+2)
#     for row in range(16):
#         f_one = fig.add_subplot(gs[row, 1])
#         img1 = images[16+row].permute([1, 2, 0])
#         f_one.imshow(img1)
#         f_one.axis("off")
#         f_two = fig.add_subplot(gs[row, 2])
#         img2 = torch.tensor(imgs[998][16+row]).permute([1, 2, 0])
#         # img = numpy.array((img - img.min()) / (img.max() - img.min()), dtype=numpy.uint8)
#         img2 = numpy.array((img2 + 1.0) * 255 / 2, dtype=numpy.uint8)
#         f_two.imshow(img2)
#         f_two.axis("off")
#         f_three = fig.add_subplot(gs[row, 3])
#         img3 = torch.tensor(imgs[998][16+row]).permute([1, 2, 0])
#         img3 = numpy.array(((img3 - img3.min()) / (img3.max() - img3.min())) * 255, dtype=numpy.uint8)
#         # img = numpy.array((img + 1.0) * 255 / 2, dtype=numpy.uint8)
#         f_three.imshow(img3)
#         f_three.axis("off")
#     plt.show()

####################################################################################################
# 计算FID
####################################################################################################
def gen_fid_input(config,unet_model,diffusion,img_num=100):
    # 获取real图像
    if config['type'] == "cifar10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        datasets.CIFAR10.url = "https://ai-studio-online.bj.bcebos.com/v1/8cf77ffb4c584eaaa716edb69eb0af6541eb532ddc0f4d00bfd7a06b113a2441?responseContentDisposition=attachment%3Bfilename%3Dcifar-10-python.tar.gz&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2025-01-23T15%3A41%3A37Z%2F21600%2F%2F8ba5a4006db020fa30e061cb18f8f7e93d5d5fce2492c17ac37c4d0f9fd7dcb2"
        dataset = datasets.CIFAR10(rf"{config['root_dir']}/data", train=True, download=True, transform=transform)
    elif config['type'] == "mnist":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        datasets.MNIST.mirrors = [
            "https://dufs.v-v.icu/mnist/",
        ]
        dataset = datasets.MNIST(rf"{config['root_dir']}/data", train=True, download=True, transform=transform)
    sampler = RandomSampler(dataset, replacement=False)
    for i in range(img_num):
        random_idx = next(iter(sampler))
        image, label = dataset[random_idx]

        save_image(image, rf"{config['root_dir']}/data/fid/real/real_{i}.png")

    # 获取生成图像
    n = 1
    while n <= img_num:
        generated_images = diffusion.sample(unet_model, config['image_size'], batch_size=64, channels=config['channel'])
        for i in range(64):
            if n <= img_num:
                img = transforms.ToTensor()((generated_images[-1][i].transpose([1, 2, 0]) + 1) / 2)
                save_image(img, rf"{config['root_dir']}/data/fid/gen/gen_{n}.png")
                n += 1
            if n % (img_num/10) == 0:
                print(f"已生成{n}个图像")

def calc_fid(real_img_dir,gen_img_dir):
    # compute FID
    # real_img_dir = r"./data/CIFAR_REAL"
    # gen_img_dir = r"./data/CIFAR_sigmoid"
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=real_img_dir,
        input2=gen_img_dir,
        cuda=True,
        fid=True,
        kid=True,
        isc=True,
        verbose=True,
    )
    print(metrics_dict)



















