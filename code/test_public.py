from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from utils import StairDetectionMobileNet, CustomDataset, transform

TEST_DIR = './stair/public/'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建自定义数据集对象
test_dataset = CustomDataset('./stair/public/public_test.json', transform=transform)


# 创建测试集的数据加载器
batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 初始化模型
model = StairDetectionMobileNet().to(DEVICE)

# 加载训练好的模型参数
model.load_state_dict(torch.load("./result/best_model.pt"))

# 在测试集上进行推理
model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    all_predictions = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy:.4f}")


# 把all_predictions中的预测结果对应地存储到test_dataset的“prediction”中
test_set_prediction = []
for i in range(len(test_dataset)):
    # 将元组转换为列表进行修改
    temp_data_list = list(test_dataset.data[i])
    # 修改元素
    temp_data_list[2] = all_predictions[i]
    # 将列表转换回元组
    test_dataset.data[i] = tuple(temp_data_list)
    test_set_prediction.append({'dir':  test_dataset.data[i][0],  'label':  test_dataset.data[i][1], 'prediction':int(test_dataset.data[i][2])})

# 把test_dataset的数据保存到public_test.json中
with open('./stair/public/public_test.json', 'w') as f:
    json.dump(test_set_prediction, f, indent=4)

# 计算Precision和Recall
print("Calculating Precision - Recall Curve...")
precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
average_precision = average_precision_score(all_labels, all_predictions)
disp = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=average_precision)
disp.plot()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('./result/public_pr_curve.png')

# 计算mAP
print(f"mAP: {average_precision:.4f}")

