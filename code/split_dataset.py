'''
以下代码参考了大作业说明文件中的附录A：数据集切分与索引文件建立示例
'''
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
THU_DIR = './stair/tsinghua/'
SEED = 123
FILE_PREFIX = 'public'
IMG_PREFIX = 'img_public'
SPLIT_RATIO = (7, 1, 2)

# 划分数据集
random.seed(SEED)
sum_ratio = sum(SPLIT_RATIO)
normed_ratio = [ratio / sum_ratio for ratio in SPLIT_RATIO]
cusum = 0.
cdf = [0] * len(normed_ratio)
for i, ratio in enumerate(normed_ratio):
    cusum += ratio
    cdf[i] = cusum

train_set, valid_set, test_set = [], [], []
for img_dir in Path(DATA_DIR).rglob('*.jpg'):
    p = random.random()
    img_dir = str(img_dir)
    label = 0 if 'no_stairs' in img_dir else 1
    if p < cdf[0]:
        train_set.append({'dir':  img_dir,  'label':  label})
    elif p < cdf[1]:
        valid_set.append({'dir':  img_dir,  'label':  label})
    else:
        test_set.append({'dir':  img_dir,  'label':  label, 'prediction':999})

tsinghua_test_set = []
for img_dir in Path(THU_DIR).rglob('*.jpg'):
    img_dir = str(img_dir)
    label = 0 if 'no_stairs' in img_dir else 1
    tsinghua_test_set.append({'dir':  img_dir,  'label':  label, 'prediction':999})

# 将数据集划分结果保存为 JSON 文件
JSON_DIR = os.path.join(DATA_DIR, FILE_PREFIX)
THU_JSON_DIR = os.path.join(THU_DIR, 'thu')
with open(f'{JSON_DIR}_train.json',  'wt') as f:
    json.dump(train_set, f, indent=4)

with open(f'{JSON_DIR}_valid.json',  'wt') as f:
    json.dump(valid_set, f, indent=4)

with open(f'{JSON_DIR}_test.json',  'wt') as f:
    json.dump(test_set, f, indent=4)

with open(f'{THU_JSON_DIR}_test.json',  'wt') as f:
    json.dump(tsinghua_test_set, f, indent=4)
