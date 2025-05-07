import argparse
import os
import random
from multiprocessing import Process, Queue
from multiprocessing import Pool
import torch
import torchvision
from sympy.stats.sampling.sample_numpy import numpy
from torchvision import datasets
import torchvision.transforms as transforms
from torch_fidelity import calculate_metrics

from src.evaluate.evaluate import plot_subprocess, fid_subprocess
from src.evaluate.mp_evaluate_v1 import evaluate, multi_evaluate
from src.unet import UNetModel
from src.utils import import_config, plot_images, show_8_images_12_denoising_steps, show_64_images
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import logging

from src.diffusions.gamma_diffusion import GammaDiffusion
from src.diffusions.gaussian_diffusion import GaussianDiffusion
from src.diffusions.binomial_diffusion import BinomialDiffusion
from src.diffusions.negative_binomial_diffusion import NBinomialDiffusion
from src.diffusions.possion_diffusion import PossionDiffusion
from src.diffusions.optimize_gamma_diffusion_v1 import OGammaDiffusion


logging.basicConfig(level=logging.INFO)

####################################################################################################
# 训练
####################################################################################################

def train(config):
    """
    Train
    :param config:
    :return:
    """
    logging.info("Start Train...")
    root_dir = config['root_dir']
    batch_size = config['model_config']['batch_size']
    timesteps = config['model_config']['timesteps']
    beta_schedule = config['model_config']['beta_schedule']
    datasets_type = config['datasets_type']
    epochs = config['epochs']
    group_epoch = config['group_epoch']  # 多少个epoch 为一组 ，用于记录
    RESUME = True if config['resume']=="True" else False
    diffusion_type=config['diffusion_type']
    exper_type = config['exper_type']
    exper_name = f"{config['exper_type']}_{config['diffusion_type']}_{config['datasets_type']}_{config['epochs']}_{config['resume']}_{config['exper_num']}"
    eval_subprocess = config['eval_subprocess']


    diffusion_dict = {
        "Binomial":BinomialDiffusion,
        "NBinomial":NBinomialDiffusion,
        "Gaussian":GaussianDiffusion,
        "Gamma": GammaDiffusion,
        "Possion":PossionDiffusion,
        "OGamma":OGammaDiffusion,
    }
    if diffusion_type in diffusion_dict.keys() :
        diffusion = diffusion_dict[diffusion_type](timesteps=timesteps, beta_schedule=beta_schedule)

    # 设置训练设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.debug(f"Already used <{device}>")

    # 加载数据集
    if datasets_type == "cifar10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        datasets.CIFAR10.url = "https://ai-studio-online.bj.bcebos.com/v1/8cf77ffb4c584eaaa716edb69eb0af6541eb532ddc0f4d00bfd7a06b113a2441?responseContentDisposition=attachment%3Bfilename%3Dcifar-10-python.tar.gz&authorization=bce-auth-v1%2F5cfe9a5e1454405eb2a975c43eace6ec%2F2025-01-23T15%3A41%3A37Z%2F21600%2F%2F8ba5a4006db020fa30e061cb18f8f7e93d5d5fce2492c17ac37c4d0f9fd7dcb2"
        dataset = datasets.CIFAR10(rf"{root_dir}/data", train=True, download=True, transform=transform)
        dataset_channel = 3
        dataset_image_size = 32
    elif datasets_type == "mnist":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        datasets.MNIST.mirrors = [
            "https://dufs.v-v.icu/mnist/",
            "https://ossci-datasets.s3.amazonaws.com/mnist/"
        ]
        dataset = datasets.MNIST(rf"{root_dir}/data", train=True, download=True, transform=transform)
        dataset_channel = 1
        dataset_image_size = 32
    elif datasets_type == "celebA":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(178),  # 先裁剪成正方形
            # transforms.Resize(128),  # 缩放到128×128
            transforms.Resize(64),  # 缩放到64×64
            transforms.ToTensor(),  # 必须启用，将PIL图像转为Tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        dataset = datasets.CelebA(
            rf"{root_dir}/data",
            split="train",
            transform=transform,
            download=True,
        )
        dataset_channel = 3
        dataset_image_size = 64

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建unet_model
    unet_model = UNetModel(
        in_channels=dataset_channel,
        model_channels=128,
        out_channels=dataset_channel,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(2,),
        dropout=0.1
    ).float()

    # 设置优化器
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=2e-4)

    # 是否继续训练
    start_epoch = 0

    if RESUME:
        checkpoint_path = config['checkpoint_path']
        check_point = torch.load(checkpoint_path)
        unet_model.load_state_dict(check_point["model_state_dict"])
        optimizer.load_state_dict(check_point["optimizer_state_dict"])
        start_epoch = check_point["epoch"]

    # 统一张量的device
    unet_model = unet_model.to(device)

    # Unet 训练 噪声
    for epoch in range(start_epoch + 1, epochs + 1):
        print(f"即将训练 第{epoch}轮")

        for step, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            batch_size = images.shape[0]
            images = images.to(device)
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            loss = diffusion.train_losses(unet_model, images, t)
            writer.add_scalar('Batch_Loss/train', loss.item(), epoch * len(train_loader) + step)
            loss.backward()
            optimizer.step()
        if epoch % group_epoch == 0 or epoch == 1:
            check_point = {
                "epoch": epoch,
                "model_state_dict": unet_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(check_point, rf"{root_dir}/checkpoints/{exper_name}_{epoch}.pth")

            config['checkpoint_path'] = rf"{root_dir}/checkpoints/{exper_name}_{epoch}.pth"
            config['ok_epoch'] = epoch
            q.put(config)


if __name__ == '__main__':
    # 获取命令行参数
    parser = argparse.ArgumentParser(description="这是一个训练脚本，用于获取训练。")
    parser.add_argument('--config', type=str, required=True, help='输入配置文件的路径')

    args = parser.parse_args()

    if args.config:
        config_path = args.config
    else:
        config_path = rf"/configs/local_train.json"

    train_config = import_config(config_path)
    logs_dir = train_config['logs_dir']
    # experiment_name = train_config["experiment_name"]
    experiment_name = f"{train_config['exper_type']}_{train_config['diffusion_type']}_{train_config['datasets_type']}_{train_config['epochs']}_{train_config['resume']}_{train_config['exper_num']}"

    q = Queue()

    # 开启两个evaluate进程
    p = Process(target=multi_evaluate, args=(q,))
    p2 = Process(target=multi_evaluate, args=(q,))
    p.start()
    p2.start()

    # tensorboar 记录
    writer = SummaryWriter(rf'{logs_dir}/{experiment_name}',flush_secs=120)
    # 开始训练
    train(train_config)
    # 结尾工作
    writer.close()

    q.put(None)
    q.put(None)

    p.join()