import argparse
import os

import torch
import torch_fidelity
from sympy.stats.sampling.sample_numpy import numpy

from src.gamma_diffusion import GammaDiffusion
from src.gaussian_diffusion import GaussianDiffusion
from src.nb_diffusion import NBDiffusion
from src.unet import UNetModel
from src.utils import import_config

import matplotlib.pyplot as plt

####################################################################################################
# 画图
####################################################################################################
# 画图

# 画n*n张图
def paint_images_1(generated_images,img_num=8):
    # 画图
    fig = plt.figure(figsize=(img_num, img_num), constrained_layout=True)
    gs = fig.add_gridspec(img_num, img_num)
    imgs = generated_images[990].reshape(1, img_num*img_num, 1, 32, 32)
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


# 随机生成n*n张图
def generate_image(nb_diffusion,model,img_num=8):
    # shape (time_steps,batch_size,channels,image_size,image_size)
    # shape (1000,128,1,32,32)
    generated_images = nb_diffusion.sample(model, 32, batch_size=img_num*img_num, channels=1)
    paint_images_1(generated_images,img_num=img_num)


def paint_images_2(imgs,img_num=8,time_steps=1000):
    # 画图
    fig = plt.figure(figsize=(img_num, img_num+2), constrained_layout=True)
    gs = fig.add_gridspec(img_num, img_num+2)
    t_idx = [x*(time_steps//img_num) for x in range(img_num)]
    t_idx[-1] = time_steps - 21
    t_idx.append(time_steps-11)
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
    t_idx[-1] = time_steps - 21
    t_idx.append(time_steps-11)
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



####################################################################################################
# 评价
####################################################################################################

def evaluate(config):
    root_dir = evaluate_config['root_dir']
    logs_dir = evaluate_config['logs_dir']
    timesteps = config['model_config']['timesteps']
    beta_schedule = config['model_config']['beta_schedule']
    checkpoint_path = config['checkpoint_path']
    diffusion_type = config['diffusion_type']

    # 设置训练设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if diffusion_type == "Gamma":
        diffusion = GammaDiffusion(timesteps=timesteps, beta_schedule=beta_schedule)
    elif diffusion_type == "NB":
        diffusion = NBDiffusion(timesteps=timesteps, beta_schedule=beta_schedule)
    elif diffusion_type == "Gaussian":
        diffusion = GaussianDiffusion(timesteps=timesteps, beta_schedule=beta_schedule)
    elif diffusion_type == "OldNB":
        diffusion = NBDiffusion(timesteps=timesteps, beta_schedule=beta_schedule)
    else:
        diffusion = None
        ValueError("没有这个diffusion类型")

    # 加载模型
    unet_model = UNetModel(
        in_channels=1,
        model_channels=128,
        out_channels=1,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(2,),
        dropout=0.1
    ).float()
    check_point = torch.load(checkpoint_path)
    unet_model.load_state_dict(check_point["model_state_dict"])
    unet_model = unet_model.to(device)
    # generate_image(diffusion,unet_model,img_num=3)
    images = torch.tensor(diffusion.sample(unet_model, 32, batch_size=64, channels=1))
    # 画图
    paint_images_3(images, img_num=8)




if __name__ == '__main__':
    # 获取命令行参数
    parser = argparse.ArgumentParser(description="这是一个评估脚本，用于评估模型。")
    parser.add_argument('--config', type=str, required=True, help='输入配置文件的路径')
    args = parser.parse_args()

    if args.config:
        config_path = args.config
    else:
        config_path = rf"C:\Users\31409\PycharmProjects\Multi_Diffusion\configs\local_evaluate_config.json"
    evaluate_config = import_config(config_path)
    evaluate(evaluate_config)