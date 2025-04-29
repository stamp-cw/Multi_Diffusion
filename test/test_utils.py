import os.path
import unittest
from random import randint
import pprint

import matplotlib.pyplot as plt
import torch
from idna import check_nfc
from setuptools.command.setopt import config_file

from src.utils import export_config, import_config, plot_images


####################################################################################################
# 测试utils.py文件中的函数
####################################################################################################
class MyTestCase(unittest.TestCase):
    pp = pprint.PrettyPrinter(indent=4)

    @unittest.skip
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    @unittest.skip
    def test_exprot_config(self):
        rand_int = randint(1,10)
        config_path = rf"C:\Users\31409\PycharmProjects\NB_Diffusioon\configs\config_{rand_int}.json"
        config = {
            "root_dir":rf"C:\Users\31409\PycharmProjects\NB_Diffusioon"
        }

        export_config(config,config_path)

        self.assertEqual(os.path.exists(config_path),True,msg=rf"配置文件未导出")

    def test_import_config(self):
        config_path = "C:\\Users\\31409\PycharmProjects\\NB_Diffusioon\\configs\\train_config.json"
        config_dict = import_config(config_path)
        self.assertIsNotNone(config_dict, msg="导入的配置字典为空")
        self.pp.pprint(config_dict)
        self.assertIsNotNone(config_dict['model_config'], msg="model_config为空")
        self.pp.pprint(config_dict['model_config'])

    def test_plot_images(self):
        # loaded_data = torch.load('./images_pt/NB_old_images.pt')
        loaded_data = torch.load('./images_pt/NB_old_images.pt')
        images = torch.tensor(loaded_data['images'])
        self.assertIsNotNone(images)
        fig = plot_images(images,img_num=8)
        plt.show()

    def test_nb_distribution(self):
        noise = torch.distributions.NegativeBinomial(30,0.5).sample(torch.Size((8,8)))
        self.assertIsNotNone(noise)
        print(noise)
        mean = torch.mean(noise)
        variance = torch.var(noise)
        print(f"Mean: {mean}, Variance: {variance}")

    def test_normal_distribution(self):
        noise = torch.distributions.Normal(0,1).sample(torch.Size((8, 8)))
        self.assertIsNotNone(noise)
        print(noise)

    def test_gama_distribution(self):
        noise = torch.distributions.Gamma(30, 0.001).sample(torch.Size((8, 8)))
        self.assertIsNotNone(noise)
        print(noise)

if __name__ == '__main__':
    unittest.main()
