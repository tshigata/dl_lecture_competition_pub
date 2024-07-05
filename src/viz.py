import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# # データを読み込む
# train_X = torch.load('data/train_X.pt')

# # データの形状を確認
# print(train_X.shape)

# # 最初のサンプルを取得
# first_sample = train_X[0]

# # データの形状を確認
# print(first_sample.shape)

# # ファイルに保存
# torch.save(first_sample, 'data/first_sample.pt')

# first_sampleをファイルから読み込む
first_sample = torch.load('data/first_sample.pt')

# # データを可視化
# plt.figure(figsize=(10, 10))
# plt.imshow(first_sample, cmap='gray')
# plt.title('First Sample of train_X')
# plt.colorbar()
# plt.show()

# 最初の脳波データを取得
# ここでは、first_sampleが2次元または3次元の配列であると仮定し、最初のチャンネルのデータを取得します。
first_channel_data = first_sample[25]

# first_channel_dataの波形を可視化
plt.figure(figsize=(10, 5))
plt.plot(first_channel_data)
plt.title('First Channel Data of the First Sample')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()


from scipy import signal


# STFTを適用
# frequencies, times, Zxx = signal.stft(first_channel_data, fs=1.0)

# # スペクトログラム（絶対値）を可視化
# plt.pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud')
# plt.title('STFT Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.colorbar(label='Magnitude')
# plt.show()



# # 各チャンネルのデータに対してSTFTを適用し、結果をプロット
# fig, axs = plt.subplots(num_channels, 1, figsize=(10, 2*num_channels))
# for i in range(num_channels):
#     channel_data = first_sample[i]
#     frequencies, times, Zxx = signal.stft(channel_data, fs=1.0)
#     axs[i].pcolormesh(times, frequencies, np.abs(Zxx), shading='gouraud')
#     axs[i].set_ylabel('Frequency [Hz]')
#     axs[i].set_title(f'Channel {i+1}')

# axs[-1].set_xlabel('Time [sec]')
# plt.tight_layout()
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

# PyTorchテンソルをNumPy配列に変換
first_sample_np = first_sample.numpy()

# 画像を周波数領域に変換
F = fft2(first_sample_np)
F = fftshift(F)

# バンドパスフィルタの設計
rows, cols = first_sample_np.shape
crow, ccol = rows // 2, cols // 2

# バンドの数を指定
num_bands = 10

# 等間隔のバンドを生成
bands = [(i*rows//num_bands, (i+1)*rows//num_bands) for i in range(num_bands)]

# フィルタの適用とプロット
plt.figure(figsize=(10, 5))
for i, band in enumerate(bands):
    # バンドパスフィルタの作成
    mask = np.zeros((rows, cols), np.uint8)
    mask[band[0]:band[1], band[0]:band[1]] = 1

    # フィルタの適用
    F_filtered = F * mask
    f_filtered = np.abs(ifftshift(ifft2(F_filtered)))

    # プロット
    plt.subplot(1, len(bands), i+1)
    plt.imshow(f_filtered, cmap='gray')
    plt.title(f'Band {i+1}: {band[0]}-{band[1]} Hz')
    plt.axis('off')

plt.show()