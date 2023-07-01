import torch
import torch.nn as nn
import json
from torchvision import models
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm
from thop import profile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# 定义教师模型
teacher_model = models.resnet50(pretrained=True)
teacher_model.fc = nn.Linear(2048, 2)  # 假设输出类别为2

# 定义学生模型
student_model = models.mobilenet_v2(pretrained=True)
student_model.classifier[1] = nn.Linear(1280, 2)  # 假设输出类别为2

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


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=0.001)
student_optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

# 定义训练函数
def train(model, optimizer, dataloader):
    model.train()
    total_loss = 0.0
    total_correct = 0

    for images, labels in dataloader:
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()

    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = total_correct / len(dataloader.dataset)

    return epoch_loss, epoch_acc

# 定义评估函数
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()

    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = total_correct / len(dataloader.dataset)

    return epoch_loss, epoch_acc

# 将教师模型的预测结果作为学生模型的训练目标
def distillation_loss(student_outputs, teacher_outputs, labels, temperature=3.0):
    soft_labels = nn.functional.softmax(teacher_outputs / temperature, dim=1)
    return nn.functional.kl_div(nn.functional.log_softmax(student_outputs / temperature, dim=1),
                                soft_labels, reduction='batchmean')

# 在训练集上训练教师模型
print("Training teacher model...")
teacher_model = teacher_model.to(device)
teacher_model.train()
torch.save(teacher_model.state_dict(), './result/distillation/teacher_model.pt')

for epoch in tqdm(range(num_epochs)):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        teacher_optimizer.zero_grad()
        outputs = teacher_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        teacher_optimizer.step()

torch.save(teacher_model.state_dict(), './result/distillation/teacher_model.pt')
print("Teacher model trained.")

# 使用教师模型的输出指导学生模型的训练
print("Training student model...")
student_model = student_model.to(device)
student_model.train()
torch.save(student_model.state_dict(), './result/distillation/student_model.pt')

for epoch in tqdm(range(num_epochs)):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        student_optimizer.zero_grad()
        teacher_outputs = teacher_model(images)
        student_outputs = student_model(images)

        loss = distillation_loss(student_outputs, teacher_outputs, labels)
        loss.backward()
        student_optimizer.step()

torch.save(student_model.state_dict(), './result/distillation/student_model.pt')
print("Student model trained.")

# 在测试集上评估学生模型的性能
print("Evaluating student model...")
student_model.eval()
test_loss, test_acc = evaluate(student_model, valid_loader)

print(f"Student Model Test Loss: {test_loss:.4f}")
print(f"Student Model Test Accuracy: {test_acc:.4f}")

print("Calculating FLOPS")
# 计算学生模型的FLOPs
input_size = (1, 3, 224, 224)
flops = torch.cuda.FloatTensor(1).zero_()
student_model = student_model.to(device)
flops += profile(student_model, inputs=(torch.zeros(input_size).to(device),))[0]
print(f"Student Model FLOPs: {flops.item()}")
