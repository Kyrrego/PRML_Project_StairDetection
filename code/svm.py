import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, PrecisionRecallDisplay
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from PIL import Image
import numpy as np
import json
import random
from pathlib import Path
import os

DATA_DIR = './stair/public/'
SEED = 123
FILE_PREFIX = 'public'
IMG_PREFIX = 'img_public'
SPLIT_RATIO = (7, 1, 2)

# 从json文件中读取数据集
print("Loading dataset...")
IMG_JSON_DIR = os.path.join(DATA_DIR, IMG_PREFIX)
# 读取训练集的JSON文件
with open(f'{IMG_JSON_DIR}_train.json', 'r') as f:
    # 用tqmd进度条显示读取的进度
    train_set = json.load(f)


# 读取验证集的JSON文件
with open(f'{IMG_JSON_DIR}_valid.json', 'r') as f:
    valid_set = json.load(f)

# 读取测试集的JSON文件
with open(f'{IMG_JSON_DIR}_test.json', 'r') as f:
    test_set = json.load(f)

print("Dataset loaded.")


# 将图像向量和标签准备为训练集和测试集的形式
X_train = np.array([data['image'] for data in train_set])
y_train = np.array([data['label'] for data in train_set])
X_test = np.array([data['image'] for data in test_set])
y_test = np.array([data['label'] for data in test_set])

X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

# TODO: 自选SVM的参数（kernel，C等）


# 创建SVM分类器并进行训练
print("Training SVM...")
svm = SVC(kernel = 'rbf', C=10, probability=True)

svm.fit(X_train, y_train)


# 在验证集上进行预测
y_pred = svm.predict(X_test)

# 计算模型的准确率和混淆矩阵
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

# 打印模型的准确率和混淆矩阵
print("Training completed.")
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion)

# 画出输出分类分数方法的Precision - Recall 曲线mAP,并把曲线保存下来
print("Calculating Precision - Recall Curve...")
prob_pred = svm.predict_proba(X_test)[:, 1]
average_precision = average_precision_score(y_test, prob_pred)
precision, recall, _ = precision_recall_curve(y_test, prob_pred)
disp = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=average_precision)
disp.plot()
plt.savefig('./result/svm_pr_curve.png')

# 计算mAP
print(f"mAP: {average_precision:.4f}")


