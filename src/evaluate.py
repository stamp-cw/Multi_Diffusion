import argparse
import os

import torch
#import torch_fidelity
from sympy.stats.sampling.sample_numpy import numpy
import torchvision.transforms as transforms
from src.unet import UNetModel
from src.utils import import_config, gen_fid_input
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
    paint_images_3(images, img_num)
    return 'ok'

def fid_subprocess(config,unet_model,diffusion,img_num=100):
    gen_fid_input(config,unet_model,diffusion,img_num)
    calc_fid(rf"{config['root_dir']}/data/fid/real",rf"{config['root_dir']}/data/fid/gen")
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

    # 执行eval子流程
    subprocess_dict = {
        'plot':{'func':plot_subprocess,'args':(
            {'type': datasets_type, 'image_size': dataset_image_size, 'channel': dataset_channel,'root_dir':root_dir},
            unet_model,
            diffusion,
            8
        )},
        'fid':{
            'func':fid_subprocess,
            'args':(
                {'type':datasets_type,'image_size':dataset_image_size,'channel':dataset_channel,'root_dir':root_dir},
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
    evaluate(evaluate_config)