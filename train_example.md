# OGamma扩散模型训练配置说明

本文档详细说明了 `featurize_train_gamma_celeba.json` 配置文件中各个参数的含义和用途。

## 基础配置

| 参数 | 值 | 说明 |
|------|-----|------|
| root_dir | /home/featurize/Multi_Diffusion | 项目根目录路径 |
| logs_dir | /home/featurize/Multi_Diffusion/logs | 日志文件保存路径 |
| diffusion_type | OGamma | 扩散模型类型：使用OGamma扩散模型 |
| experiment_name | train_OGamma_celeba_001 | 实验名称：用于标识不同的训练运行 |
| epochs | 100 | 训练轮数：总共训练100轮 |
| resume | False | 是否从检查点恢复训练 |
| checkpoint_path | "" | 检查点路径：如果resume为true，则从此路径加载模型 |
| group_epoch | 1 | 分组训练轮数：每1轮进行一次评估 |
| datasets_type | celeba | 数据集类型：使用CelebA数据集 |

## 模型配置 (model_config)

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 64 | 训练批次大小：每批处理64张图片 |
| timesteps | 1000 | 扩散步数：噪声添加和去除的总步数 |
| beta_schedule | linear | beta调度方式：使用线性调度 |
| image_size | 64 | 输入图像大小：64x64像素 |
| in_channels | 3 | 输入通道数：RGB三通道 |
| hidden_size | 64 | 基础隐藏层大小：64个通道 |
| num_res_blocks | 2 | 残差块数量：每个分辨率使用2个残差块 |
| attention_resolutions | [16, 8] | 注意力机制的分辨率：在16x16和8x8分辨率上使用注意力 |
| dropout | 0.1 | Dropout比率：防止过拟合 |
| channel_mult | [1, 2, 2, 2] | 通道倍增因子：控制网络宽度 |
| num_heads | 4 | 注意力头数：多头注意力机制 |
| num_head_channels | 64 | 每个注意力头的通道数 |
| resblock_updown | true | 是否在上下采样时使用残差连接 |
| use_new_attention_order | true | 是否使用新的注意力顺序 |

## 数据集配置 (dataset_config)

| 参数 | 值 | 说明 |
|------|-----|------|
| data_dir | data/celeba | 数据集根目录 |
| image_size | 64 | 图像大小：64x64像素 |
| center_crop | true | 是否进行中心裁剪 |
| random_flip | true | 是否进行随机水平翻转（数据增强） |
| train_batch_size | 64 | 训练集批次大小 |
| val_batch_size | 64 | 验证集批次大小 |
| num_workers | 4 | 数据加载的工作进程数 |

## 优化器配置 (optimizer_config)

| 参数 | 值 | 说明 |
|------|-----|------|
| learning_rate | 1e-4 | 学习率 |
| weight_decay | 0.0 | 权重衰减：L2正则化系数 |
| beta1 | 0.9 | Adam优化器的beta1参数 |
| beta2 | 0.999 | Adam优化器的beta2参数 |
| scheduler | cosine | 学习率调度器：使用余弦退火 |
| warmup_steps | 5000 | 预热步数：前5000步进行学习率预热 |

## 保存配置 (save_config)

| 参数 | 值 | 说明 |
|------|-----|------|
| save_every | 5000 | 每5000步保存一次模型 |
| save_best | true | 是否保存最佳模型 |
| max_keep | 5 | 最多保留5个检查点 |

## 日志配置 (logging_config)

| 参数 | 值 | 说明 |
|------|-----|------|
| log_every | 100 | 每100步记录一次日志 |
| log_images | true | 是否记录生成的图像 |
| log_fid | true | 是否记录FID分数 |
| log_metrics | ["loss", "fid"] | 需要记录的指标：损失和FID分数 |

## 使用说明

1. 确保CelebA数据集已下载并放在 `data/celeba` 目录下
2. 使用以下命令开始训练：
```bash
python src/train.py --config configs/featurize_train_gamma_celeba.json
```

## 注意事项

1. 训练过程可能需要较长时间，建议使用GPU
2. 模型会自动保存检查点，可以随时恢复训练
3. 可以通过修改 `save_config` 和 `logging_config` 来调整保存和日志频率
4. 如果显存不足，可以适当减小 `batch_size` 和 `