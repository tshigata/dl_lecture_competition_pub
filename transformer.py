import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
import numpy as np
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import KFold
import os
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm
from termcolor import cprint

from src.preprocess import CosineScheduler
from src.preprocess import EarlyStopping

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import KFold
import torchvision.models as models
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torchmetrics import Accuracy
import os
import time
import datetime
import pytz

from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from timm.scheduler import CosineLRScheduler
from tqdm import tqdm
from termcolor import cprint

from src.preprocess import CosineScheduler
from src.preprocess import EarlyStopping
from src.preprocess import preprocess_eeg_data
from src.preprocess import AugmentedSubset
from src.preprocess import AugmentedDataset

from src.models import EEGNet
from src.models import ShallowConvNet
from src.models import DeepConvNet
from src.models import ResNet18
from src.models import LSTMModel
from src.models import CNNLSTM
from src.models import DenseNet
from src.models import GRUModel
from src.models import InceptionNet
from src.utils import set_seed

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb

# ダミーデータ生成関数
def create_dummy_data(num_samples=1000, num_channels=271, num_timepoints=281, num_classes=1854):
    X_train = torch.randn(num_samples, 1, num_channels, num_timepoints)
    y_train = torch.randint(0, num_classes, (num_samples,))
    subject_idxs_train = torch.randint(0, 4, (num_samples,))
    return X_train, y_train, subject_idxs_train

# データセットの定義
class EEGDataset(Dataset):
    def __init__(self, X, y=None, subject_ids=None):
        self.X = X
        self.y = y
        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx], self.subject_ids[idx]
        else:
            return self.X[idx], self.subject_ids[idx]
# 1エポック分の処理を関数化
def train_and_validate_one_epoch(epoch, model, train_loader, val_loader, optimizer, scheduler, accuracy, device, args):
    print(f"Epoch {epoch+1}/{args.num_epochs}")

    train_loss, train_acc, val_loss, val_acc = [], [], [], []

    # トレーニングフェーズ
    model.train()
    for X, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} Training'):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = F.cross_entropy(y_pred, y)
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = accuracy(y_pred, y)
        train_acc.append(acc.item())

    # 検証フェーズ
    model.eval()
    for X, y in tqdm(val_loader, desc="Validation"):
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            y_pred = model(X)

        val_loss.append(F.cross_entropy(y_pred, y).item())
        val_acc.append(accuracy(y_pred, y).item())
    
    scheduler.step(np.mean(val_loss))  # 学習率を調整
    # エポックの結果を表示
    print(f"Epoch {epoch+1}/{args.num_epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")

    return np.mean(train_loss), np.mean(train_acc), np.mean(val_loss), np.mean(val_acc)

def train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, early_stopping, accuracy, device, args, save_folder_name, fold=None):
    max_val_acc = 0

    for epoch in range(args.num_epochs):
        train_loss, train_acc, val_loss, val_acc = train_and_validate_one_epoch(epoch, model, train_loader, val_loader, optimizer, scheduler, accuracy, device, args)
        if args.use_wandb:
            wandb.log({
                f'loss/train/fold-{fold}' if fold is not None else 'loss/train': train_loss,
                f'acc/train/fold-{fold}' if fold is not None else 'acc/train': train_acc,
                f'loss/validation/fold-{fold}' if fold is not None else 'loss/validation': val_loss,
                f'acc/validation/fold-{fold}' if fold is not None else 'acc/validation': val_acc,
                'epoch': epoch
            })
        if epoch == 0:
            torch.save(model.state_dict(), os.path.join(save_folder_name, f"model_best_fold{fold+1}.pt" if fold is not None else "model_best.pt"))
            print(f"Initial model for Fold {fold+1} saved." if fold is not None else "Initial model saved.")

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_folder_name, f"model_best_fold{fold+1}.pt" if fold is not None else "model_best.pt"))
            cprint(f"New best. Max Val Acc = {max_val_acc:.5f}", "cyan")

        if args.use_wandb:
            wandb.log({f'max_val_acc/validation/fold-{fold}' if fold is not None else 'max_val_acc/validation': max_val_acc})

        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            print("Early stopping. Max Acc = ", max_val_acc)
            break

    return max_val_acc


# # Transformerエンコーダモデルの定義
# class EEGTransformerEncoder(nn.Module):
#     def __init__(self, input_dim, nhead, num_encoder_layers, dim_feedforward, dropout, num_subjects, embedding_dim, num_classes):
#         super(EEGTransformerEncoder, self).__init__()
#         self.subject_embedding = nn.Embedding(num_subjects, embedding_dim)
#         combined_dim = input_dim + embedding_dim
#         assert combined_dim % nhead == 0, "combined_dim must be divisible by nhead"
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=combined_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
#         self.fc = nn.Linear(combined_dim, num_classes)

#     def forward(self, src, subject_ids):
#         # srcとsubject_embedsのサイズを確認
#         print("src size:", src.size())
#         subject_embeds = self.subject_embedding(subject_ids).unsqueeze(1).expand(-1, src.size(1), -1)
#         src = src.squeeze(1)  # 形状を (batch_size, 271, 128) に変更
#         print("src size after squeeze:", src.size())
#         print("subject_embeds size:", subject_embeds.size())
#         src = torch.cat((src, subject_embeds), dim=2)  # 形状を (batch_size, 271, 136) に変更
#         src = self.transformer_encoder(src)
#         src = src.mean(dim=1)
#         output = self.fc(src)
#         return output


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
        return F.log_softmax(x, dim=1)
        # return x
    
# メイン関数
@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    # ハイパーパラメータの設定
    input_dim = cfg.dataset.num_timepoints
    nhead = 8
    num_encoder_layers = 6
    dim_feedforward = 512
    dropout = cfg.training.dropout_rate
    num_subjects = 4
    embedding_dim = 8
    num_classes = cfg.dataset.num_classes
    device = torch.device(cfg.training.device)

    if cfg.training.use_dryrun:
        # ダミーデータを生成
        X_train, y_train, subject_idxs_train = create_dummy_data()
        print(X_train.shape)
        print(y_train.shape)
        print(subject_idxs_train.shape)
    else:
        data_dir = cfg.dataset.data_dir
        # 本番データを読み込み
        train_X = torch.load(f"{cfg.dataset.data_dir}/train_X.pt")
        train_y = torch.load(f"{cfg.dataset.data_dir}/train_y.pt")
        val_X = torch.load(f"{cfg.dataset.data_dir}/val_X.pt")
        val_y = torch.load(f"{cfg.dataset.data_dir}/val_y.pt")
        train_subject_idxs = torch.load(os.path.join(data_dir, 'train_subject_idxs.pt'))
        val_subject_idxs = torch.load(os.path.join(data_dir, 'val_subject_idxs.pt'))

        X_train = torch.cat((train_X, val_X), dim=0)
        y_train = torch.cat((train_y, val_y), dim=0)
        subject_idxs_train = torch.cat((train_subject_idxs, val_subject_idxs), dim=0)

        #X_train, y_train, subject_idxs_train の形状をprint
        print(X_train.shape)
        print(y_train.shape)
        print(subject_idxs_train.shape)

    # モデルのインスタンス化
    # ハイパーパラメータの設定
    num_timepoints = cfg.dataset.num_timepoints
    num_channels = cfg.dataset.num_channels
    nhead = 8

    # d_modelをnheadで割り切れるように調整
    product = num_channels * num_timepoints
    if product % nhead != 0:
        d_model = (product // nhead + 1) * nhead
    else:
        d_model = product

    num_encoder_layers = 6
    dim_feedforward = 512
    dropout = cfg.training.dropout_rate
    dropout = 0.1
    num_subjects = 4
    embedding_dim = 8
    num_classes = cfg.dataset.num_classes
    device = torch.device(cfg.training.device)

    model = EEGTransformerEncoder(num_classes=num_classes, num_channels=num_channels, num_timepoints=num_timepoints)
    model.to(device)

    # オプティマイザの設定
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    accuracy = Accuracy(
    task="multiclass", num_classes=cfg.dataset.num_classes, top_k=10
    ).to(device)

    # データセットとデータローダーの定義
    dataset = EEGDataset(X_train, y_train, subject_idxs_train)
    kf = KFold(n_splits=10, shuffle=True, random_state=cfg.training.seed)

    for train_index, val_index in kf.split(dataset):
        train_subset = torch.utils.data.Subset(dataset, train_index)
        val_subset = torch.utils.data.Subset(dataset, val_index)
        train_loader = DataLoader(train_subset, batch_size=cfg.other.batch_size, shuffle=True, num_workers=cfg.training.num_workers)
        val_loader = DataLoader(val_subset, batch_size=cfg.other.batch_size, shuffle=False, num_workers=cfg.training.num_workers)

        scheduler = CosineLRScheduler(optimizer, t_initial=100, lr_min=1e-6,
                                      warmup_t=3, warmup_lr_init=1e-6, warmup_prefix=True)
        early_stopping = EarlyStopping(patience=2, verbose=True)

        max_val_acc = train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, early_stopping, accuracy, device)

        cprint(f"Fold max Acc = {max_val_acc}", "cyan")
        cprint("------------------------", "cyan")

        # # トレーニングと評価関数の定義
        # def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
        #     for epoch in range(num_epochs):
        #         # トレーニング
        #         model.train()
        #         running_loss = 0.0
        #         correct_train = 0
        #         total_train = 0
        #         for inputs, labels, subject_ids in train_loader:
        #             inputs, labels, _ = inputs.to(device), labels.to(device), subject_ids.to(device)

        #             optimizer.zero_grad()
        #             # outputs = model(inputs, subject_ids)
        #             outputs = model(inputs)
        #             loss = criterion(outputs, labels)
        #             loss.backward()
        #             optimizer.step()

        #             running_loss += loss.item() * inputs.size(0)
        #             # 予測と正解の比較
        #             _, preds = torch.max(outputs, 1)
        #             correct_train += (preds == labels).sum().item()
        #             total_train += labels.size(0)

        #         epoch_loss = running_loss / len(train_loader.dataset)
        #         train_accuracy = correct_train / total_train

        #         # バリデーション
        #         model.eval()
        #         running_val_loss = 0.0
        #         correct_val = 0
        #         total_val = 0
        #         with torch.no_grad():
        #             for inputs, labels, subject_ids in val_loader:
        #                 inputs, labels, _ = inputs.to(device), labels.to(device), subject_ids.to(device)
        #                 # outputs = model(inputs, subject_ids)
        #                 outputs = model(inputs)
        #                 loss = criterion(outputs, labels)
        #                 running_val_loss += loss.item() * inputs.size(0)
        #                 # 予測と正解の比較
        #                 _, preds = torch.max(outputs, 1)
        #                 correct_val += (preds == labels).sum().item()
        #                 total_val += labels.size(0)

        #         val_loss = running_val_loss / len(val_loader.dataset)
        #         val_accuracy = correct_val / total_val

        #         print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # # モデルのトレーニング
        # train_model(model, train_loader, val_loader, criterion, optimizer, cfg.other.num_epochs)

        break  # 1回の分割だけで終了

# 実行
if __name__ == "__main__":
    main()
