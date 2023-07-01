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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
            self.data.append((image_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image, label

# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3), # 将灰度图像转换为3个通道的图像
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 设置训练参数
batch_size = 32
learning_rate = 0.001
num_epochs = 10

# 创建训练集、验证集和测试集的自定义数据集对象
train_dataset = CustomDataset('./stair/public/public_train.json', transform=transform)
valid_dataset = CustomDataset('./stair/public/public_valid.json', transform=transform)
test_dataset = CustomDataset('./stair/public/public_test.json', transform=transform)

# 创建训练集、验证集和测试集的数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型和优化器
model = StairDetectionMobileNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_valid_accuracy = 0.0  # 保存最佳验证集准确率

for epoch in tqdm(range(num_epochs)):
    # 训练
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 验证
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = total_correct / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}")

        # 如果当前验证集准确率更高，则保存模型
        if accuracy > best_valid_accuracy:
            best_valid_accuracy = accuracy
            torch.save(model.state_dict(), "./result/best_model.pt")

# 加载最佳模型参数
model.load_state_dict(torch.load("./result/best_model.pt"))

# 在测试集上进行推理
model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0

    all_predictions = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = total_correct / total_samples
    # 计算Precision和Recall
    precision, recall, _ = precision_recall_curve(all_labels, all_predictions)

    print(f"Test Accuracy: {accuracy:.4f}")

# 绘制Precision-Recall曲线
plt.figure()
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.show()
# 保存曲线
plt.savefig('./result/mobilenet_pr_curve.png')

# 计算mAP
average_precision = average_precision_score(all_labels, all_predictions)
print(f"mAP: {average_precision:.4f}")