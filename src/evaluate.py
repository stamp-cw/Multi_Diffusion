import argparse
import json
import os

import torch
import torchvision
#import torch_fidelity
from tensorboardX import SummaryWriter
from sympy.stats.sampling.sample_numpy import numpy
import torchvision.transforms as transforms
from src.unet import UNetModel
from src.utils import import_config, gen_fid_input, show_64_images, show_8_images_12_denoising_steps, \
    show_8_images_raw_and_denoise, get_raw_images
from torchvision import datasets
import matplotlib.pyplot as plt

from src.utils import generate_image,paint_images_1,paint_images_2,paint_images_3
from src.utils import calc_fid

from src.gamma_diffusion import GammaDiffusion
from src.gaussian_diffusion import GaussianDiffusion
from src.binomial_diffusion import BinomialDiffusion
from src.negative_binomial_diffusion import NBinomialDiffusion
from src.possion_diffusion import PossionDiffusion
from src.optimize_gamma_diffusion import OGammaDiffusion

####################################################################################################
# 评价
####################################################################################################

def plot_subprocess(config,unet_model,diffusion,img_num):

    images = torch.tensor(diffusion.sample(unet_model, config['image_size'], batch_size=64, channels=config['channel']))
    fig_one = show_64_images(images, step=1000)
    writer.add_figure(rf"show_64_images.png", fig_one)

    fig_two = show_8_images_12_denoising_steps(images)
    writer.add_figure(rf"show_8_images_12_denoising_steps.png", fig_two)

    raw_images = get_raw_images(config)
    denoise_images = diffusion.sampleA(unet_model, 32, raw_images, 128, 1)


    fig_three = show_8_images_raw_and_denoise(raw_images, denoise_images, step=1000)
    writer.add_figure(rf"show_8_images_raw_and_denoise.png", fig_three)

    raw_images_grid = torchvision.utils.make_grid(raw_images)
    writer.add_image('raw_images_grid', raw_images_grid, 0)

    # 过渡图
    for step in [0, 50, 100, 200, 400, 600, 800, 900, 970, 990, 998, 999]:
        random_images_grid = torchvision.utils.make_grid(images[step])
        denoise_images_grid = torchvision.utils.make_grid(denoise_images[step])
        writer.add_image('random_images_grid', random_images_grid, step)
        writer.add_image('denoise_images_grid', denoise_images_grid, step)

    return 'ok'

def fid_subprocess(config,unet_model,diffusion,img_num=100):
    gen_fid_input(config,unet_model,diffusion,img_num)
    metrics_dict = calc_fid(rf"{config['root_dir']}/data/fid/{config['exper_name']}/real",rf"{config['root_dir']}/data/fid/{config['exper_name']}/gen")
    writer.add_scalar(rf'fid/evaluate', metrics_dict['frechet_inception_distance'], 100)
    writer.add_text("fid_metrics_dict", json.dumps(metrics_dict, indent=2), global_step=0)
    return 'ok'

def evaluate(config):
    # 加载变量与模型
    root_dir = evaluate_config['root_dir']
    logs_dir = evaluate_config['logs_dir']
    timesteps = config['model_config']['timesteps']
    beta_schedule = config['model_config']['beta_schedule']
    checkpoint_path = config['checkpoint_path']
    datasets_type = config['datasets_type']
    diffusion_type = config['diffusion_type']
    eval_subprocess = config['eval_subprocess']
    # exper_name = f"{config['diffusion_type']}_{config['datasets_type']}_{config['epochs']}"
    exper_name = f"{config['experiment_name']}"
    batch_size = config['model_config']['batch_size']


    # 设置训练设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    diffusion_dict = {
        "Binomial":BinomialDiffusion,
        "NBinomial":NBinomialDiffusion,
        "Gaussian":GaussianDiffusion,
        "Gamma": GammaDiffusion,
        "Possion":PossionDiffusion,
        "OGamma":OGammaDiffusion,
    }
    if diffusion_type in diffusion_dict.keys():
        diffusion = diffusion_dict[diffusion_type](timesteps=timesteps, beta_schedule=beta_schedule)
    else:
        diffusion = None
        ValueError("没有这个diffusion类型")

    dataset_dict = {
        'mnist': {
            'dataset_channel':1,
            'dataset_image_size':32
        },
        'cifar10':{
            'dataset_channel': 3,
            'dataset_image_size': 32
        }
    }

    if datasets_type in dataset_dict.keys():
        dataset_channel = dataset_dict[datasets_type]['dataset_channel']
        dataset_image_size = dataset_dict[datasets_type]['dataset_image_size']
    else:
        dataset_channel = None
        dataset_image_size = None
        ValueError("没有这个dataset类型")

    # 加载模型
    unet_model = UNetModel(
        in_channels=dataset_channel,
        model_channels=128,
        out_channels=dataset_channel,
        channel_mult=(1, 2, 2, 2),
        attention_resolutions=(2,),
        dropout=0.1
    ).float()
    check_point = torch.load(checkpoint_path)
    unet_model.load_state_dict(check_point["model_state_dict"])
    unet_model = unet_model.to(device)
    # generate_image(diffusion,unet_model,img_num=3)
    # images = torch.tensor(diffusion.sample(unet_model, dataset_image_size, batch_size=64, channels=dataset_channel))

    # writer.add_graph(unet_model, torch.randn(batch_size,dataset_channel,dataset_image_size,dataset_image_size))

    # 执行eval子流程
    subprocess_dict = {
        'plot':{'func':plot_subprocess,'args':(
            {'type': datasets_type, 'image_size': dataset_image_size, 'channel': dataset_channel,'root_dir':root_dir,'exper_name':exper_name},
            unet_model,
            diffusion,
            8
        )},
        'fid':{
            'func':fid_subprocess,
            'args':(
                {'type':datasets_type,'image_size':dataset_image_size,'channel':dataset_channel,'root_dir':root_dir,'exper_name':exper_name},
                unet_model,
                diffusion,
                100
            )
        },
    }

    subprocess_status_list = []

    for subprocess in eval_subprocess:
        if subprocess in subprocess_dict:
            func_info = subprocess_dict[subprocess]
            status = func_info['func'](*func_info['args'])
            subprocess_status_list.append({'subprocess':subprocess,'status':status})

if __name__ == '__main__':
    # 获取命令行参数
    parser = argparse.ArgumentParser(description="这是一个评价脚本，用于评价模型。")
    parser.add_argument('--config', type=str, required=True, help='输入配置文件的路径')
    args = parser.parse_args()

    if args.config:
        config_path = args.config
    else:
        config_path = rf"D:\Project\Multi_Diffusion\configs\local_evaluate.json"
    evaluate_config = import_config(config_path)
    logs_dir = evaluate_config['logs_dir']
    experiment_name = evaluate_config["experiment_name"]

    # tensorboar 记录
    writer = SummaryWriter(rf'{logs_dir}/{experiment_name}')
    evaluate(evaluate_config)
    # 结尾工作
    writer.close()