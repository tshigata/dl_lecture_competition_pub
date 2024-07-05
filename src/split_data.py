import torch
import numpy as np
from datasets import ThingsMEGDataset

# .ptファイルからデータを読み込む
train_subject_idxs = torch.load('data/train_subject_idxs.pt')
val_subject_idxs = torch.load('data/val_subject_idxs.pt')
test_subject_idxs = torch.load('data/test_subject_idxs.pt')

print(train_subject_idxs.shape)
print(train_subject_idxs.unique())
print(train_subject_idxs.bincount())
print(train_subject_idxs.bincount().argmax())
#先頭の100個の要素を表示
print(train_subject_idxs[:100])

print(val_subject_idxs.shape)
print(val_subject_idxs.unique())
print(val_subject_idxs.bincount())
print(val_subject_idxs.bincount().argmax())
#先頭の100個の要素を表示
print(val_subject_idxs[:100])

print(test_subject_idxs.shape)
print(test_subject_idxs.unique())
print(test_subject_idxs.bincount())
print(test_subject_idxs.bincount().argmax())
#先頭の100個の要素を表示
print(test_subject_idxs[:100])


# subject_idxsの値が変化する位置を取得
train_change_idxs = np.where(np.diff(train_subject_idxs) != 0)[0] + 1
val_change_idxs = np.where(np.diff(val_subject_idxs) != 0)[0] + 1
test_change_idxs = np.where(np.diff(test_subject_idxs) != 0)[0] + 1

# ThingsMEGDatasetクラスのインスタンス化
# ここでは仮にtrain, val, testデータセットが存在すると仮定します
train_dataset = ThingsMEGDataset('train')
val_dataset = ThingsMEGDataset('val')
test_dataset = ThingsMEGDataset('test')

print(train_dataset.X.shape)
print(train_dataset.y.shape)
print(train_dataset.subject_idxs.shape)

print(val_dataset.X.shape)
print(val_dataset.y.shape)
print(val_dataset.subject_idxs.shape)

print(test_dataset.X.shape)
print(test_dataset.subject_idxs.shape)

# 初期インデックスを設定
start_idx_train = start_idx_val = start_idx_test = 0

import matplotlib.pyplot as plt

# subject_idxsの値に基づいてデータを分割し、個別のファイルに保存
for i in range(4):
    # 分割位置を取得
    end_idx_train = train_change_idxs[i] if i < len(train_change_idxs) else len(train_subject_idxs)
    end_idx_val = val_change_idxs[i] if i < len(val_change_idxs) else len(val_subject_idxs)

    # データを分割
    train_y_i = train_dataset.y[start_idx_train:end_idx_train]
    val_y_i = val_dataset.y[start_idx_val:end_idx_val]

    # データを保存
    torch.save(train_y_i, f'data/{i}_train_y.pt')
    torch.save(val_y_i, f'data/{i}_val_y.pt')

    # ヒストグラムを作成
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(train_y_i, bins=range(1855), alpha=0.7, color='blue', label='train')
    plt.title(f'Class Distribution of Train Dataset {i}')
    plt.xlabel('Class')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(val_y_i, bins=range(1855), alpha=0.7, color='orange', label='val')
    plt.title(f'Class Distribution of Validation Dataset {i}')
    plt.xlabel('Class')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    # 次の分割のためにインデックスを更新
    start_idx_train = end_idx_train
    start_idx_val = end_idx_val