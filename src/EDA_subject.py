import torch
import matplotlib.pyplot as plt

# .ptファイルからデータを読み込む
train_subject_idxs = torch.load('data/train_subject_idxs.pt')
val_subject_idxs = torch.load('data/val_subject_idxs.pt')
test_subject_idxs = torch.load('data/test_subject_idxs.pt')

print("Train dataset:")
print(train_subject_idxs.shape)
print(train_subject_idxs.unique())
print(train_subject_idxs.bincount())
print(train_subject_idxs.bincount().argmax())
#先頭の100個の要素を表示
print(train_subject_idxs[:100])
#先頭から16340番目から16350番目までの要素を表示
print(train_subject_idxs[16430:16450])

print("Validation dataset:")
print(val_subject_idxs.shape)
print(val_subject_idxs.unique())
print(val_subject_idxs.bincount())
print(val_subject_idxs.bincount().argmax())
#先頭の100個の要素を表示
print(val_subject_idxs[:100])

print("Test dataset:")
print(test_subject_idxs.shape)
print(test_subject_idxs.unique())
print(test_subject_idxs.bincount())
print(test_subject_idxs.bincount().argmax())
#先頭の100個の要素を表示
print(test_subject_idxs[:100])

import numpy as np

# subject_idxsの値が変化する位置を調べる
train_change_idxs = np.where(np.diff(train_subject_idxs) != 0)[0] + 1
val_change_idxs = np.where(np.diff(val_subject_idxs) != 0)[0] + 1
test_change_idxs = np.where(np.diff(test_subject_idxs) != 0)[0] + 1

print("Train dataset subject change indices:")
print(train_change_idxs)
print("\nValidation dataset subject change indices:")
print(val_change_idxs)
print("\nTest dataset subject change indices:")
print(test_change_idxs)

# data/test_Xを100個表示
test_X = torch.load('data/test_X.pt')
print("Test dataset X:")
print(test_X.shape)
print(test_X[:100])

# data/train_Xを100個表示
train_X = torch.load('data/train_X.pt')
print("Train dataset X:")
print(train_X.shape)
print(train_X[:100])


# trainとvalを合わせたデータセットを作成。ただし、subject_idxsが昇順になるように並べ替える

