# Multi-Diffusion: 多模态扩散模型实现

## 项目简介
本项目实现了多种扩散模型（Diffusion Models）的变体，包括：
- Gamma 扩散模型
- 标准扩散模型
- 其他扩散模型变体

## 项目结构
```
.
├── configs/           # 配置文件目录
│   ├── featurize_train_gamma.json    # Gamma模型训练配置
│   └── featurize_train_standard.json # 标准扩散模型训练配置
├── src/              # 源代码目录
├── test/             # 测试代码目录
├── checkpoints/      # 模型检查点保存目录
├── logs/            # 训练日志目录
├── data/            # 数据集目录
├── resource/        # 资源文件目录
└── scripts/         # 脚本文件目录
```

## 环境要求
- Python 3.8+
- PyTorch 1.8+
- 其他依赖见 requirements.txt

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 训练模型
#### Gamma 扩散模型
```bash
python src/train.py --config configs/featurize_train_gamma.json
```

#### 标准扩散模型
```bash
python src/train.py --config configs/featurize_train_standard.json
```

### 3. 配置说明
配置文件包含以下主要参数：
- `diffusion_type`: 扩散模型类型（"Gamma" 或 "standard"）
- `model_config`: 模型配置
  - `batch_size`: 批次大小
  - `timesteps`: 扩散步数
  - `beta_schedule`: beta调度方式
- `epochs`: 训练轮数
- `datasets_type`: 数据集类型
- `eval_subprocess`: 评估方法

## 模型说明

### Gamma 扩散模型
Gamma 扩散模型使用 Gamma 分布来建模噪声过程，相比标准扩散模型具有以下特点：
- 更灵活的噪声分布
- 可能在某些任务上具有更好的性能
- 支持不同的 beta 调度策略

### 标准扩散模型
标准扩散模型使用高斯分布作为噪声过程，是扩散模型的基础实现。

## 评估指标
- FID (Fréchet Inception Distance)
- 生成样本可视化
- 其他自定义评估指标

## 注意事项
1. 训练前请确保有足够的计算资源
2. 建议使用 GPU 进行训练
3. 可以通过修改配置文件调整模型参数

## 引用
如果您使用了本项目的代码，请引用以下论文：
```text
@article{song2020denoising,
  title={},
  author={},
  journal={},
  year={},
  month={},
  abbr={},
  url={}
}
```

## 许可证
[待定]

## 联系方式
[待定]
