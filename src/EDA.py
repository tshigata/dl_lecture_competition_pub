# import pandas as pd
# import matplotlib.pyplot as plt
from datasets import ThingsMEGDataset

# # ThingsMEGDatasetクラスのインスタンス化
# # ここでは仮にtrain, val, testデータセットが存在すると仮定します
# train_dataset = ThingsMEGDataset('train', data_dir='data')
# val_dataset = ThingsMEGDataset('val', data_dir='data')
# test_dataset = ThingsMEGDataset('test', data_dir='data')

import pandas as pd
import matplotlib.pyplot as plt

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

# yの値を取得
train_y = train_dataset.y
val_y = val_dataset.y

#train_yの値を先頭から100個表示
print("train_yの値を先頭から100個表示")
print(train_y.shape)
print(train_y[:100])
#val_yの値を先頭から100個表示
print(val_y.shape)
print(val_y[:100])


# # ヒストグラムを作成
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.hist(train_y, bins=range(1855), alpha=0.7, color='blue', label='train')
# plt.title('Class Distribution of Train Dataset')
# plt.xlabel('Class')
# plt.ylabel('Frequency')

# plt.subplot(1, 2, 2)
# plt.hist(val_y, bins=range(1855), alpha=0.7, color='orange', label='val')
# plt.title('Class Distribution of Validation Dataset')
# plt.xlabel('Class')
# plt.ylabel('Frequency')

# plt.tight_layout()
# plt.show()

# # データセットをpandas DataFrameに変換
# train_X_df = pd.DataFrame(train_dataset.X)
# train_y_df = pd.DataFrame(train_dataset.y)
# val_X_df = pd.DataFrame(val_dataset.X)
# val_y_df = pd.DataFrame(val_dataset.y)
# test_X_df = pd.DataFrame(test_dataset.X)

# # 基本的な統計情報を表示
# print("Train data (X):")
# print(train_X_df.describe())
# print("Train data (y):")
# print(train_y_df.describe())
# print("\nValidation data (X):")
# print(val_X_df.describe())
# print("Validation data (y):")
# print(val_y_df.describe())
# print("\nTest data (X):")
# print(test_X_df.describe())

# # データの分布をヒストグラムで表示
# train_X_df.hist(figsize=(10, 10))
# plt.show()

# train_y_df.hist(figsize=(10, 10))
# plt.show()

# val_X_df.hist(figsize=(10, 10))
# plt.show()

# val_y_df.hist(figsize=(10, 10))
# plt.show()

# test_X_df.hist(figsize=(10, 10))
# plt.show()

# #train_Xの1個目のデータを2次元表示
# plt.figure(figsize=(10, 5))
# plt.plot(train_dataset.X[0].T)
# plt.title('Train Dataset X[0]')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.show()

#train_Xは、271x281の脳波波形です。1枚のデータを縦に積み上げて、一度に表示したいです。1個目のデータを使い、脳波波形を縦に並べて表示
plt.figure(figsize=(10, 5))
plt.plot(train_dataset.X[0].T)
plt.title('Train Dataset X[0]')
plt.xlabel('Time')






