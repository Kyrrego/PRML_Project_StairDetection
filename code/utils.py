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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt

DATA_DIR = './stair/public/'
THU_DIR = './stair/tsinghua/'
SEED = 123
FILE_PREFIX = 'public'
IMG_PREFIX = 'img_public'
SPLIT_RATIO = (7, 1, 2)

# 定义MobileNet模型
class StairDetectionMobileNet(nn.Module):
    def __init__(self):
        super(StairDetectionMobileNet, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(1280, 2)

    def forward(self, x):
        x = self.model(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, json_path, transform=None):
        self.data = []
        self.transform = transform

        with open(json_path, 'r') as f:
            data_list = json.load(f)

        for data in data_list:
            image_path = data['dir']
            label = data['label']
            prediction = data.get('prediction', 999)  # 获取prediction字段的值，如果不存在则设置为999
            self.data.append((image_path, label, prediction))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label, prediction = self.data[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label, prediction, image_path




# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3), # 将灰度图像转换为3个通道的图像
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])