import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.svm import SVC
from PIL import Image
import numpy as np
import json
import random
from pathlib import Path
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt

from utils import StairDetectionMobileNet, CustomDataset, transform

DATA_DIR = './stair/public/'
THU_DIR = './stair/tsinghua/'
SEED = 123
FILE_PREFIX = 'public'
IMG_PREFIX = 'img_public'
SPLIT_RATIO = (7, 1, 2)

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
print(device)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# 设置训练参数
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# 创建训练集、验证集和测试集的自定义数据集对象
train_dataset = CustomDataset('./stair/public/public_train.json', transform=transform)
valid_dataset = CustomDataset('./stair/public/public_valid.json', transform=transform)
test_dataset = CustomDataset('./stair/public/public_test.json', transform=transform)

# 创建训练集、验证集和测试集的数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None

    def backward_hook(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def generate_heatmap(self, input_image, target_class):
        # 设置反向传播钩子
        hook = self.target_layer.register_backward_hook(self.backward_hook)

        # 前向传播
        logits = self.model(input_image)
        self.model.zero_grad()

        # 计算目标类的梯度
        one_hot = torch.zeros_like(logits)
        one_hot[0][target_class] = 1
        logits.backward(gradient=one_hot)

        # 计算特征图权重
        weights = torch.mean(self.gradient, dim=(2, 3))
        weights = torch.clamp(weights, min=0)
        weights /= torch.sum(weights)

        # 生成热度图
        with torch.no_grad():
            outputs = self.model.model.features(input_image)
        heatmap = torch.zeros(outputs.size(2), outputs.size(3)).to(device)
        for i, weight in enumerate(weights[0]):
            heatmap += weight * outputs[0][i, :, :]

        # 调整热度图的大小，使其与输入图像的大小一致
        heatmap = heatmap.cpu().numpy()
        heatmap = cv2.resize(heatmap, (image.width, image.height))


        # 移除钩子
        hook.remove()

        return heatmap


# 加载预训练的MobileNet模型
model = StairDetectionMobileNet().to(device)
model.load_state_dict(torch.load("./result/best_model.pt"))
model.eval()

# 选择目标层（通常是最后一个卷积层）
target_layer = model.model.features[-1]

# 创建Grad-CAM实例
grad_cam = GradCAM(model, target_layer)

# 选择要可视化的图像索引
# public_true: './stair/public/stairs/down/IMG_20190513_130745.jpg'
# public_false: 'stair/public/no_stairs/IMG_20190513_171945.jpg'
# tsinghua_true: 'stair/tsinghua/scene3/stairs/frame0204.jpg'
# tsinghua_false: 'stair/tsinghua/scene4/stairs/frame0017.jpg'

image_index = 0
image_path = 'stair/tsinghua/scene4/stairs/frame0017.jpg'

# 加载测试图像: open image in image_path
image = Image.open(image_path).convert("RGB")
label = 0


# 将图像转换为PyTorch张量并添加一个维度
input_image = transforms.ToTensor()(image).unsqueeze(0)

# 将图像转换为RGB格式
#input_image = input_image.repeat(1, 3, 1, 1)

# 将图像张量移动到设备（CPU或GPU）
input_image = input_image.to(device)

# 选择要可视化的目标类
target_class = 1  # 假设为有楼梯的类别

# 生成热度图
heatmap = grad_cam.generate_heatmap(input_image, target_class)

# 将热度图进行归一化处理
heatmap = cv2.resize(heatmap, (image.width, image.height))
heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
heatmap = np.uint8(255 * heatmap)


# heatmap转换成RGB格式
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 打印image和heatmap的形状
print('Image shape:', image.size)
print('Heatmap shape:', heatmap.shape)
print('Image:', np.array(image).shape)


# 叠加热度图到原始图像
overlay = cv2.addWeighted(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), 0.7, heatmap, 0.3, 0)

# 显示原始图像和叠加后的图像
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title('Heatmap Overlay')
plt.axis('off')
plt.savefig('./result/grad_CAM/tsinghua_false.png')

#plt.show()
