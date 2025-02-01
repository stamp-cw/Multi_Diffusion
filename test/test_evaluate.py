import unittest

import torch

from src.evaluate import paint_images_1, paint_images_2, paint_images_3
from src.gamma_diffusion import GammaDiffusion
from src.nb_diffusion import NBDiffusion
from src.old_nb_diffusion import OLDNBDiffusion
from src.unet import UNetModel
from src.utils import import_config


class MyTestCase(unittest.TestCase):
    @unittest.skip
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_save_image_tensor(self):

        diffusion_type="OldNB"

        if diffusion_type=="Gamma":
            config_path = rf"C:\Users\31409\PycharmProjects\Multi_Diffusion\configs\local_evaluate_gamma_config.json"
        elif diffusion_type == "NB":
            # config_path = rf"C:\Users\31409\PycharmProjects\Multi_Diffusion\configs\local_evaluate_nb_config.json"
            config_path = rf"C:\Users\31409\PycharmProjects\Multi_Diffusion\configs\local_evaluate_nb_config.json"
            # config_path = rf"C:\Users\31409\PycharmProjects\Multi_Diffusion\configs\old_server_evaluate_nb_config.json"
        elif diffusion_type == "Gaussian":
            pass
        elif diffusion_type == "OldNB":
            config_path = rf"C:\Users\31409\PycharmProjects\Multi_Diffusion\configs\local_evaluate_old_nb_config.json"

        # self.assertIsNotNone(diffusion_type)
        print(config_path)

        config = import_config(config_path)

        timesteps = config['model_config']['timesteps']
        beta_schedule = config['model_config']['beta_schedule']
        checkpoint_path = config['checkpoint_path']

        # 设置训练设备
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # nb_diffusion = NBDiffusion(timesteps=timesteps, beta_schedule=beta_schedule)
        diffusion = GammaDiffusion(timesteps=timesteps, beta_schedule=beta_schedule)
        # diffusion = OLDNBDiffusion(timesteps=timesteps, beta_schedule=beta_schedule)

        # 加载模型
        unet_model = UNetModel(
            in_channels=3,
            model_channels=128,
            out_channels=3,
            channel_mult=(1, 2, 2, 2),
            attention_resolutions=(2,),
            dropout=0.1
        ).float()
        check_point = torch.load(checkpoint_path)
        unet_model.load_state_dict(check_point["model_state_dict"])
        unet_model = unet_model.to(device)
        generated_images = diffusion.sample(unet_model, 32, batch_size=64, channels=3)
        self.assertIsNotNone(generated_images)
        torch.save({
            "images":generated_images
        },f"./images_pt/{diffusion_type}_test_003_images.pt")

    def test_paint_image_1(self):
        loaded_data = torch.load('./images_pt/images.pt')
        images = loaded_data['images']
        self.assertIsNotNone(images)
        paint_images_1(images)

    def test_paint_image_2(self):
        loaded_data = torch.load('./images_pt/images.pt')
        images = torch.tensor(loaded_data['images'])
        self.assertIsNotNone(images)
        paint_images_2(images)

    def test_paint_image_3(self):
        # loaded_data = torch.load('./images_pt/NB_old_images.pt')
        # loaded_data = torch.load('./images_pt/NB_old_images.pt')
        loaded_data = torch.load('./images_pt/OldNB_test_003_images.pt')
        images = torch.tensor(loaded_data['images'])
        self.assertIsNotNone(images)
        paint_images_3(images,img_num=8)





if __name__ == '__main__':
    unittest.main()
