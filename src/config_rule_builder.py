root_dir_list=[
    "/home/featurize/Multi_Diffusion",
]
# logs_dir

diffusion_type_list=[
    "Binomial",
    "NBinomial",
    "Gaussian",
    "Gamma",
    "Possion",
    "OGamma",
]
exper_number=[
    "001",
    "002",
    "003",
]

batch_size_list=[
    128,
]

timesteps_list=[
    1000,
]

beta_schedule_list=[
    "linear",
]

epochs_list=[
    10,
    20,
    50,
    100,
    200,
    400,
    1000,
]

resume_list=[
    "Ture",
    "False",
]
# checkpoint_path

# exper_name_list
group_epoch_list=[
    1,
    10,
    20,
    50,
    100,
]

datasets_type_list=[
    "mnist",
    "cifar10",
    "celebA",
]
# eval_subprocess


# 按字段列表生成
def build_by_filed_list(filed_list,example_config_dict):
    for filed_item in filed_list:
        pass
    pass

def complete_and_check_by_dict():
    '''
    dec: complete compose filed in dict
    '''
    pass

def complete_and_check_by_file():
    '''
    dec: complete compose filed in file
    '''
    pass

def complete_and_check_by_dir():
    '''
    dec: complete compose filed in all file
    '''
    pass