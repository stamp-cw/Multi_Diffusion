import argparse
from src.utils import export_config

config_dict = {
    'root_dir': '/home/featurize/Multi_Diffusion',
    'logs_dir': '/home/featurize/Multi_Diffusion/logs',
    'diffusion_type': 'Gaussian',
    'experiment_name': 'train_Gaussian_mnist_1_001',
    'model_config': {
        'batch_size': 128,
        'timesteps': 1000,
        'beta_schedule': 'linear'
    },
    'epochs': 1,
    'resume': 'False',
    'checkpoint_path': '',
    'group_epoch': 1,
    'datasets_type': 'mnist',
    'exper_type': 'train',
    'exper_num': '001',
    "eval_subprocess": ["plot","fid"]
}

# 获取命令行参数
parser = argparse.ArgumentParser(description="")
parser.add_argument('--root_dir', type=str, required=False, default="/home/featurize/Multi_Diffusion",help='')
parser.add_argument('--logs_dir', type=str, required=False, default="/home/featurize/Multi_Diffusion/logs",help='')
parser.add_argument('--diffusion_type', type=str, required=False, default="Gaussian",help='')
parser.add_argument('--batch_size', type=int, required=False, default=128,help='')
parser.add_argument('--timesteps', type=int, required=False, default=1000,help='')
parser.add_argument('--beta_schedule', type=str, required=False, default="linear",help='')
parser.add_argument('--epochs', type=int, required=False, default=1,help='')
parser.add_argument('--resume', type=str, required=False, default="False",help='')
parser.add_argument('--checkpoint_path', type=str, required=False, default="",help='')
parser.add_argument('--group_epoch', type=int, required=False, default=1,help='')
parser.add_argument('--datasets_type', type=str, required=False, default="mnist",help='')
parser.add_argument('--exper_type', type=str, required=False, default="train",help='')
parser.add_argument('--exper_num', type=str, required=False, default="001",help='')
parser.add_argument('--is_gen_file', type=bool, required=False, default=False,help='')

args = parser.parse_args()

config_dict['root_dir'] = args.root_dir
config_dict['logs_dir'] = args.logs_dir
config_dict['diffusion_type'] = args.diffusion_type
config_dict['model_config']['batch_size'] = args.batch_size
config_dict['model_config']['timesteps'] = args.timesteps
config_dict['model_config']['beta_schedule'] = args.beta_schedule
config_dict['epochs'] = args.epochs
config_dict['resume'] = args.resume
config_dict['checkpoint_path'] = args.checkpoint_path
config_dict['group_epoch'] = args.group_epoch
config_dict['datasets_type'] = args.datasets_type
config_dict['exper_type'] = args.exper_type
config_dict['exper_num'] = args.exper_num
config_dict['experiment_name'] = f"{config_dict['exper_type']}_{config_dict['diffusion_type']}_{config_dict['datasets_type']}_{config_dict['epochs']}_{config_dict['exper_num']}"

if args.is_gen_file:
    export_config(config_dict,f"featurize_{config_dict['experiment_name']}.json")

print(config_dict)