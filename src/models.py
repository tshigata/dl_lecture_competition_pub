import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torchvision.models as models

# EEGNetのモデル定義
class EEGNet(nn.Module):
    def __init__(self, num_classes, Chans=271, Samples=128, dropout_rate=0.25):
        super(EEGNet, self).__init__()
        
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16)
        )
        
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (Chans, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=(1, 4)),
            nn.Dropout(dropout_rate)
        )
        
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 8)),
            nn.Dropout(dropout_rate)
        )
        
        self.classify = nn.Sequential(
            nn.Linear(32 * ((Samples // 32)), num_classes)
        )

    def forward(self, x):
        # 入力xにunsqueeze(1)を適用して、新しい次元を挿入
        # x = x.unsqueeze(1)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x

class EEGNetImproved(nn.Module):
    def __init__(self, num_classes, Chans=271, Samples=128, dropout_rate=0.5):
        super(EEGNetImproved, self).__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((1, 2))  # 適切なプーリングサイズに調整
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(32, 64, (Chans, 1), stride=(1, 1), groups=32, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),  # プーリングサイズを調整
            nn.Dropout(dropout_rate)
        )

        self.separableConv1 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),  # プーリングサイズを調整
            nn.Dropout(dropout_rate)
        )

        self.separableConv2 = nn.Sequential(
            nn.Conv2d(128, 256, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),  # プーリングサイズを調整
            nn.Dropout(dropout_rate)
        )

        # プーリング層の変更に合わせてflattened_sizeを再計算
        self.flattened_size = 256 * ((Samples // 2 // 2 // 2 // 2))  
        self.classify = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv1(x)
        x = self.separableConv2(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x

class EEGTransformerEncoder(nn.Module):
    def __init__(self, num_classes, num_channels, num_timepoints, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        """
        EEGTransformerEncoderの初期化。
        
        :param num_classes: 分類するクラスの数
        :param num_channels: 脳波データのチャンネル数
        :param num_timepoints: 各チャンネルの時間点の数
        :param d_model: 埋め込みの次元数
        :param nhead: マルチヘッドアテンションのヘッド数
        :param num_encoder_layers: エンコーダレイヤーの数
        :param dim_feedforward: フィードフォワードネットワークの次元数
        :param dropout: ドロップアウト率
        """
        super(EEGTransformerEncoder, self).__init__()
        self.num_channels = num_channels
        self.num_timepoints = num_timepoints
        self.d_model = d_model
        self.nhead = nhead

        self.embedding = nn.Linear(num_channels * num_timepoints, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.fc_out = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        EEGTransformerEncoderのフォワードパス。
        
        :param x: 入力データ。形状は(batch_size, num_channels, num_timepoints)
        """
        # 入力xの形状を(batch_size, num_channels*num_timepoints)に変更
        x = x.view(x.size(0), -1)
        # データを埋め込み空間に投影
        x = self.embedding(x)
        # 埋め込みベクトルをエンコーダに通す
        x = self.transformer_encoder(x.unsqueeze(1))
        # クラス分類のための線形層
        x = self.fc_out(x[:, 0, :])
        # return F.log_softmax(x, dim=1)
        return x


class ShallowConvNet(nn.Module):
    def __init__(self, num_classes, num_channels, input_time_length):
        super(ShallowConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 40, (1, 13), stride=1)
        self.conv2 = nn.Conv2d(40, 40, (num_channels, 1), stride=1)
        self.pool1 = nn.AvgPool2d((1, 35), stride=(1, 7))
        self.conv2_drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(40 * ((input_time_length - 13 + 1) // 7), num_classes)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.pool1(x)
        x = self.conv2_drop(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class DeepConvNet(nn.Module):
    def __init__(self, num_classes, num_channels, input_time_length):
        super(DeepConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 25, (1, 10), stride=1, padding=(0, 5))
        self.conv2 = nn.Conv2d(25, 25, (num_channels, 1), stride=1, padding=(0, 0))
        self.pool1 = nn.AvgPool2d((1, 3), stride=(1, 3))
        self.conv3 = nn.Conv2d(25, 50, (1, 10), stride=1, padding=(0, 5))
        self.pool2 = nn.AvgPool2d((1, 3), stride=(1, 3))
        self.conv4 = nn.Conv2d(50, 100, (1, 10), stride=1, padding=(0, 5))
        self.pool3 = nn.AvgPool2d((1, 3), stride=(1, 3))
        self.conv5 = nn.Conv2d(100, 200, (1, 10), stride=1, padding=(0, 5))
        self.pool4 = nn.AvgPool2d((1, 3), stride=(1, 3))

        self.flattened_size = 200 * ((input_time_length // 3 // 3 // 3 // 3))
        self.fc1 = nn.Linear(self.flattened_size, num_classes)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.pool1(x)
        x = F.elu(self.conv3(x))
        x = self.pool2(x)
        x = F.elu(self.conv4(x))
        x = self.pool3(x)
        x = F.elu(self.conv5(x))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    

class ResNet18(nn.Module):
    def __init__(self, num_classes=1854):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# class LSTMModel(nn.Module):
#     def __init__(self, num_classes=1854):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(128*2, num_classes)

#     def forward(self, x):
#         x = x.squeeze(1)  # Remove channel dimension
#         lstm_out, _ = self.lstm(x)
#         out = self.fc(lstm_out[:, -1, :])  # Use the output of the last time step
#         return out

class LSTMModel(nn.Module):
    def __init__(self, num_classes=1854, dropout_rate=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(128*2, num_classes)

    def forward(self, x):
        x = x.squeeze(1)  # Remove channel dimension
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out[:, -1, :])  # Use the output of the last time step
        return out

class CNNLSTM(nn.Module):
    def __init__(self, num_classes=1854):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=64 * 67 * 70, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128*2, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1, 64 * 67 * 70)  # Reshape for LSTM
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Use the output of the last time step
        return out

class DenseNet(nn.Module):
    def __init__(self, num_classes=1854):
        super(DenseNet, self).__init__()
        self.model = models.densenet121(pretrained=False)
        self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# class GRUModel(nn.Module):
#     def __init__(self, num_classes=1854):
#         super(GRUModel, self).__init__()
#         self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(128*2, num_classes)

#     def forward(self, x):
#         x = x.squeeze(1)  # Remove channel dimension
#         gru_out, _ = self.gru(x)
#         out = self.fc(gru_out[:, -1, :])  # Use the output of the last time step
#         return out
    
class GRUModel(nn.Module):
    def __init__(self, num_classes=1854, dropout_rate=0.5):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(128*2, num_classes)

    def forward(self, x):
        x = x.squeeze(1)  # Remove channel dimension
        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out)
        out = self.fc(gru_out[:, -1, :])  # Use the output of the last time step
        return out

class InceptionNet(nn.Module):
    def __init__(self, num_classes=1854):
        super(InceptionNet, self).__init__()
        self.model = models.inception_v3(pretrained=False, aux_logits=False)
        self.model.Conv2d_1a_3x3 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        return self.model(x)


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)
    
import random
import numpy as np
import torch
from scipy.signal import resample, butter, filtfilt

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
