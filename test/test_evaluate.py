import json
import random
import unittest

import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import RandomSampler

from src.utils import show_64_images,show_8_images_12_denoising_steps,show_8_images_raw_and_denoise
from src.optimize_gamma_diffusion_v0 import OGammaDiffusion
from src.gaussian_diffusion import GaussianDiffusion
from src.unet import UNetModel
from src.utils import import_config
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt

class MyTestCase(unittest.TestCase):

    @unittest.skip
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_save_image_tensor(self):

        config_path=rf"D:\Project\Multi_Diffusion\configs\local_evaluate_optimize_gamma.json"
        # config_path=rf"D:\Project\Multi_Diffusion\configs\local_evaluate_gaussian.json"

        config = import_config(config_path)

        timesteps = config['model_config']['timesteps']
        beta_schedule = config['model_config']['beta_schedule']
        checkpoint_path = config['checkpoint_path']
        root_dir = config['root_dir']

        datasets_type = config['datasets_type']
        diffusion_type = config['diffusion_type']

        # 设置训练设备
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if diffusion_type == "Gaussian":
            diffusion = GaussianDiffusion(timesteps=timesteps, beta_schedule=beta_schedule)
        elif diffusion_type == "OGamma":
            diffusion = OGammaDiffusion(timesteps=timesteps, beta_schedule=beta_schedule)

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
        generated_images = diffusion.sample(unet_model, 32, batch_size=64, channels=1)
        self.assertIsNotNone(generated_images)
        torch.save({
            "images":generated_images
        },f"{root_dir}/data/test/{diffusion_type}_{datasets_type}_images.pt")

    def test_paint_image_4(self):
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(padding=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        datasets.MNIST.mirrors = [
            "https://dufs.v-v.icu/mnist/",
        ]
        dataset = datasets.MNIST(rf"D:\Project\Multi_Diffusion\data", train=True, download=True, transform=transform)
        # sampler = RandomSampler(dataset, replacement=False)
        # for i in range(64):
        #     random_idx = next(iter(sampler))
        #     image, label = dataset[random_idx]

        indices = random.sample(range(len(dataset)), 64)
        real_images = torch.stack([dataset[i][0] for i in indices])
        print(real_images.shape)

        torch.save({
            "images":real_images
        },f"../data/test/real_images.pt")


    def test_show_64_images(self):
        loaded_data = torch.load('../data/test/images.pt')
        images = loaded_data['images']
        self.assertIsNotNone(images)
        show_64_images(images)
        plt.show()

    def test_show_8_images_12_denoising_steps(self):
        loaded_data = torch.load('../data/test/images.pt')
        images = torch.tensor(loaded_data['images'])
        self.assertIsNotNone(images)
        show_8_images_12_denoising_steps(images)
        plt.show()

    def test_how_8_images_raw_and_denoise(self):
        loaded_data = torch.load('../data/test/images.pt')
        real_loaded_data = torch.load('../data/test/real_images.pt')

        images = torch.tensor(loaded_data['images'])
        real_images = torch.tensor(real_loaded_data['images'])
        self.assertIsNotNone(images)
        show_8_images_raw_and_denoise(real_images,images)
        plt.show()


    def test_tensorboard(self):
        loaded_data = torch.load('../data/test/images.pt')
        images = torch.tensor(loaded_data['images'])
        real_loaded_data = torch.load('../data/test/real_images.pt')
        real_images = torch.tensor(real_loaded_data['images'])
        writer = SummaryWriter(rf'D:\Project\Multi_Diffusion\logs\test')
        # # zero
        # for step in [0, 50, 100, 200, 400, 600, 800, 900, 970, 990, 998, 999]:
        #     images_grid = torchvision.utils.make_grid(images[step])
        #     writer.add_image(rf"transition.png", images_grid, step)
        # # one
        # print(images.shape)
        # fig_one = show_64_images(images, step=1000)
        # writer.add_figure(rf"show_64_images.png", fig_one)

        # # two
        # fig_two = show_8_images_12_denoising_steps(images)
        # writer.add_figure(rf"show_8_images_12_denoising_steps.png", fig_two)

        # three
        # fig_three = show_8_images_raw_and_denoise(real_images, images, step=1000)
        # writer.add_figure(rf"show_8_images_raw_and_denoise.png", fig_three)

        # four
        # writer.add_text("fid_metrics_dict", json.dumps({'a':1,'b':2}, indent=2), global_step=0)
        writer.close()


    def test_tensorboard2(self):
        loaded_data = torch.load('../data/test/images.pt')
        images = torch.tensor(loaded_data['images'])
        real_loaded_data = torch.load('../data/test/real_images.pt')
        real_images = torch.tensor(real_loaded_data['images'])
        writer = SummaryWriter(rf'D:\Project\Multi_Diffusion\logs\test')

        unet_model = UNetModel(
            in_channels=1,
            model_channels=128,
            out_channels=1,
            channel_mult=(1, 2, 2, 2),
            attention_resolutions=(2,),
            dropout=0.1
        ).float()

        check_point = torch.load(rf"D:\Project\Multi_Diffusion\checkpoints\checkpoint_OGama_002_OGamma_False_9.pth")
        unet_model.load_state_dict(check_point["model_state_dict"])
        unet_model = unet_model.to("cuda")

        x_t = torch.randn(128,1,32,32,device="cuda")
        t = torch.full((128,), 999, dtype=torch.long,device="cuda")

        input_data = [x_t,t]

        writer.add_graph(unet_model,input_data)

        writer.close()





if __name__ == '__main__':
    unittest.main()
