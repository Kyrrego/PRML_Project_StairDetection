from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
import numpy as np
import matplotlib.pyplot as plt

THU_DIR = './stair/tsinghua/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, data_dir, scene=None, transform=None):
        self.data_dir = data_dir
        self.scene = scene
        self.transform = transform
        self.file_list = []
        self.labels = []

        # 获取文件列表和标签
        if scene is None:
            for scene in range(1, 6):
                scene_dir = os.path.join(data_dir, f'scene{scene}')
                for label in ['no_stairs', 'stairs']:
                    label_dir = os.path.join(scene_dir, label)
                    for file_name in sorted(os.listdir(label_dir)):
                        if file_name.endswith('.jpg'):
                            self.file_list.append(os.path.join(label_dir, file_name))
                            self.labels.append(0 if label == 'no_stairs' else 1)
        else:
            # scene_dir = os.path.join(data_dir, self.scene)
            scene_dir = data_dir
            for label in ['no_stairs', 'stairs']:
                label_dir = os.path.join(scene_dir, label)
                for file_name in sorted(os.listdir(label_dir)):
                    if file_name.endswith('.jpg'):
                        self.file_list.append(os.path.join(label_dir, file_name))
                        self.labels.append(0 if label == 'no_stairs' else 1)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        image_path = self.file_list[index]
        label = self.labels[index]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label

# 数据预处理和加载
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为3个通道的图像
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 创建自定义数据集对象
test_dataset = CustomDataset('./stair/tsinghua/', transform=transform)

# 创建测试集的数据加载器
batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
model = StairDetectionMobileNet().to(device)

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

# 保存预测结果
np.savetxt("./result/predictions_tsinghua.txt", all_predictions, fmt="%d")

# 在scene1～scene5中的数据上进行测试
scenes = ['scene1', 'scene2', 'scene3', 'scene4', 'scene5']
scene_accuracies = []
scene_precisions = []
scene_recalls = []

for scene in scenes:
    scene_dir = os.path.join(THU_DIR, scene)
    scene_dataset = CustomDataset(scene_dir, scene=scene, transform=transform)
    scene_loader = DataLoader(scene_dataset, batch_size=batch_size, shuffle=False)

    scene_correct = 0
    scene_total = 0
    scene_predictions = []
    scene_labels = []

    for images, labels in scene_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        scene_correct += (predicted == labels).sum().item()
        scene_total += labels.size(0)

        scene_predictions.extend(predicted.cpu().numpy())
        scene_labels.extend(labels.cpu().numpy())

    scene_accuracy = scene_correct / scene_total
    scene_precision, scene_recall, _ = precision_recall_curve(scene_labels, scene_predictions)

    scene_accuracies.append(scene_accuracy)
    scene_precisions.append(scene_precision)
    scene_recalls.append(scene_recall)

    print(f"Test Accuracy ({scene}): {scene_accuracy:.4f}")

# 计算Precision和Recall
print("Calculating Precision - Recall Curve...")
precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
average_precision = average_precision_score(all_labels, all_predictions)
disp = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=average_precision)
disp.plot()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('./result/new_pr_curve.png')

# 计算mAP
print(f"mAP: {average_precision:.4f}")