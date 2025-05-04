import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置样式
sns.set(style="whitegrid")

# 示例数据 - 替换为你的实际数据
original_data = np.random.normal(loc=0, scale=1, size=1000)  # 原始数据
generated_data = np.random.normal(loc=0.5, scale=1.2, size=1000)  # 生成数据

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制分布图
sns.kdeplot(original_data, label="Original data", fill=True, alpha=0.5)
sns.kdeplot(generated_data, label="Generated data", fill=True, alpha=0.5)

# 添加标题和标签
plt.title("DLPM, α=1.7", fontsize=14)
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()

# 显示图形
plt.show()