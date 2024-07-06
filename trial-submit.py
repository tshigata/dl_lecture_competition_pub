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
from src.models import EEGNetImproved
from src.models import EEGNetWithSubject
from src.models import EEGNetWithSubjectBatchNorm
from src.models import ShallowConvNet
from src.models import DeepConvNet
from src.models import ResNet18
from src.models import LSTMModel
from src.models import CNNLSTM
from src.models import DenseNet
from src.models import GRUModel
from src.models import InceptionNet
from src.models import EEGTransformerEncoder
from src.utils import set_seed

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb

# ダミーデータの作成関数
def create_dummy_data():
    X = torch.randn(1000, 1, 271, 128)
    y = torch.randint(0, 1854, (1000,))
    # X_test = torch.randn(100, 1, 271, 128)
    subject_idxs_train = torch.randint(0, 4, (1000,))
    return TensorDataset(X, y, subject_idxs_train)

def print_tensor_info(tensor, tensor_name):
    print(f"{tensor_name}のShape:", tensor.shape)

def load_data(data_dir, force_preprocess):
    train_X = torch.load(os.path.join(data_dir, 'train_X.pt'))
    val_X = torch.load(os.path.join(data_dir, 'val_X.pt'))
    train_y = torch.load(os.path.join(data_dir, 'train_y.pt'))
    val_y = torch.load(os.path.join(data_dir, 'val_y.pt'))
    train_subject_idxs = torch.load(os.path.join(data_dir, 'train_subject_idxs.pt'))
    val_subject_idxs = torch.load(os.path.join(data_dir, 'val_subject_idxs.pt'))

    # データの形状を確認
    print_tensor_info(train_X, "train_X")
    print_tensor_info(val_X, "val_X")
    print_tensor_info(train_y, "train_y")
    print_tensor_info(val_y, "val_y")
    print_tensor_info(train_subject_idxs, "train_subject_idxs")
    print_tensor_info(val_subject_idxs, "val_subject_idxs")

    # データの結合  
    X = torch.cat((train_X, val_X), dim=0)
    y = torch.cat((train_y, val_y), dim=0)
    subject_idxs_train = torch.cat((train_subject_idxs, val_subject_idxs), dim=0)

    print("結合後：")
    print_tensor_info(X, "X")
    print_tensor_info(y, "y")
    print_tensor_info(subject_idxs_train, "subject_idxs_train")

    # 前処理済みデータの保存パス
    preprocessed_data_path = os.path.join(data_dir, 'preprocessed_data.pt')

    if (not force_preprocess) and os.path.exists(preprocessed_data_path):
        # 前処理済みデータが存在する場合、ロードする
        X = torch.load(preprocessed_data_path)
    else:
        # 前処理を行い、保存する
        X = torch.tensor(preprocess_eeg_data(X.numpy()), dtype=torch.float32).unsqueeze(1)
        torch.save(X, preprocessed_data_path)

    dataset = TensorDataset(X, y, subject_idxs_train)

    print("前処理後：")
    print_tensor_info(X, "X")
    print_tensor_info(y, "y")
    print_tensor_info(subject_idxs_train, "subject_idxs_train")
    print("データセットのShape:", dataset[0][0].shape)

    return dataset

model_classes = {
    'EEGNet': EEGNet,
    'EEGNetImproved': EEGNetImproved,
    'EEGNetWithSubject': EEGNetWithSubject,
    'EEGNetWithSubjectBatchNorm': EEGNetWithSubjectBatchNorm,
    'ShallowConvNet': ShallowConvNet,
    'DeepConvNet': DeepConvNet,
    'ResNet18': ResNet18,
    'LSTMModel': LSTMModel,
    'CNNLSTM': CNNLSTM,
    'DenseNet': DenseNet,
    'GRUModel': GRUModel,
    'InceptionNet': InceptionNet,
    'EEGTransformerEncoder': EEGTransformerEncoder,
}

# 1エポック分の処理を関数化
def train_and_validate_one_epoch(epoch, model, train_loader, val_loader, optimizer, scheduler, accuracy, device, args):
    print(f"Epoch {epoch+1}/{args.num_epochs}")

    train_loss, train_acc, val_loss, val_acc = [], [], [], []

    # トレーニングフェーズ
    model.train()
    for X, y, subject_idxs in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.num_epochs} Training'):
        X, y, subject_idxs = X.to(device), y.to(device), subject_idxs.to(device)

        y_pred = model(X, subject_idxs)
        loss = F.cross_entropy(y_pred, y)
        train_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = accuracy(y_pred, y)
        train_acc.append(acc.item())

    # 検証フェーズ
    model.eval()
    for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
        X, y, subject_idxs = X.to(device), y.to(device), subject_idxs.to(device)

        with torch.no_grad():
            y_pred = model(X, subject_idxs)

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

def cross_validation_training(kf, dataset, ModelClass, accuracy, device, args, save_folder_name):
    max_val_acc_list = []

    for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        cprint(f'Fold {fold+1}/{args.n_splits}', "yellow")
    
        # トレーニングデータと検証データに分割
        if args.data_augmentation_train:
            train_data = AugmentedSubset(dataset, train_index, augmentation_prob=0.3)
        else:
            train_data = Subset(dataset, train_index)

        val_data = Subset(dataset, val_index)
        
        train_loader = DataLoader(train_data, batch_size=args.num_batches, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.num_batches, shuffle=False)

        # モデルの定義
        model = ModelClass(num_classes=args.num_classes).to(device)

        # 損失関数と最適化関数
        if args.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.02)
        scheduler = CosineLRScheduler(optimizer, t_initial=100, lr_min=1e-6,
                                      warmup_t=3, warmup_lr_init=1e-6, warmup_prefix=True)
        early_stopping = EarlyStopping(patience=args.num_patience, verbose=True)

        max_val_acc = train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, early_stopping, accuracy, device, args, save_folder_name, fold)
        max_val_acc_list.append(max_val_acc)

        cprint(f"Fold {fold+1} max Acc = {max_val_acc}", "cyan")
        cprint("------------------------", "cyan")

    return max_val_acc_list

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: DictConfig):
    start_time = time.time()
    set_seed(cfg.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Log directory  : {logdir}")
    save_folder_name = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if cfg.use_wandb:
        # DictConfigを通常の辞書に変換
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        wandb.init(mode="online", dir=logdir, project="MEG-classification", config=config_dict)
         
    # デバイスの設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if cfg.dry_run:
    #     dataset = create_dummy_data()
    # else:
    #     data_dir = cfg.data_dir
    #     dataset = load_data(data_dir, cfg.force_preprocess)
        
    # accuracy = Accuracy(
    #     task="multiclass", num_classes=cfg.num_classes, top_k=10
    # ).to(device)


    max_val_acc = 0
    max_val_acc_list = []

    if cfg.model_name not in model_classes:
        raise ValueError(f"モデル名 {cfg.model_name} は無効です。利用可能なモデル名: {list(model_classes.keys())}")
    
    ModelClass = model_classes[cfg.model_name]
    print("------------------------")
    print(f'Training {cfg.model_name}')

    # # 交差検証なし
    # if not cfg.use_cv:
    #     # KFoldを使ってデータセットを分割
    #     kf = KFold(n_splits=cfg.n_splits, shuffle=True)
    #     train_indices, val_indices = next(kf.split(dataset))
    
    #     cprint(f'Single fold ({cfg.n_splits-1}:{1} split)', "yellow")
    
    #     if cfg.data_augmentation_train:
    #         train_data = AugmentedDataset(Subset(dataset, train_indices), augmentation_prob=cfg.augmentation_prob)
    #     else:
    #         train_data = Subset(dataset, train_indices)
        
    #     val_data = Subset(dataset, val_indices)
    
    #     train_loader = DataLoader(train_data, batch_size=cfg.num_batches, shuffle=True)
    #     val_loader = DataLoader(val_data, batch_size=cfg.num_batches, shuffle=False)
    
    #     # モデルの定義
    #     if cfg.model_name == "EEGNet" or cfg.model_name == "EEGNetImproved":
    #         model = ModelClass(num_classes=cfg.num_classes, dropout_rate=cfg.dropout_rate).to(device)
    #     elif cfg.model_name == "EEGNetWithSubject" or cfg.model_name == "EEGNetWithSubjectBatchNorm":
    #         model = ModelClass(num_classes=cfg.num_classes, num_subjects=cfg.num_subjects, dropout_rate=cfg.dropout_rate).to(device)
    #     elif cfg.model_name == "EEGTransformerEncoder":
    #         model = EEGTransformerEncoder(num_classes=cfg.num_classes, num_channels=cfg.num_channels, num_timepoints=cfg.num_timepoints).to(device)
    #     else:
    #         model = ModelClass(num_classes=cfg.num_classes).to(device)
    
    #     # 損失関数と最適化関数
    #     if cfg.optimizer == "Adam":
    #         optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    #     else:
    #         optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.02)

    #     scheduler = CosineLRScheduler(optimizer, t_initial=100, lr_min=1e-6,
    #                                   warmup_t=3, warmup_lr_init=1e-6, warmup_prefix=True)
    #     early_stopping = EarlyStopping(patience=cfg.num_patience, verbose=True)
    
    #     max_val_acc = train_and_evaluate(model, train_loader, val_loader, optimizer, scheduler, early_stopping, accuracy, device, cfg, save_folder_name)
    #     max_val_acc_list = [max_val_acc]
        
    # else: # 交差検証あり

    #     if cfg.splitter_name == 'KFold':
    #         kf =  KFold(n_splits=cfg.n_splits, shuffle=True)
    #     else:
    #         kf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True)
    #     max_val_acc_list = cross_validation_training(kf, dataset, ModelClass, accuracy, device, cfg, save_folder_name)

    # mean_acc = np.mean(max_val_acc_list)

    # if cfg.use_wandb:
    #     wandb.log({'total_val_acc_mean': mean_acc})
    # cprint(f"Total Val Acc mean = {mean_acc} !", "cyan")

    save_folder_name = "outputs/2024-07-06/02-11-26"
    if (not cfg.dry_run) and cfg.use_cv:
        # モデルの定義

        model = ModelClass(num_classes=cfg.num_classes).to(device)


        # テストデータを読み込む
        test_X = torch.load('data/test_X.pt')
        test_subject_idxs = torch.load('data/test_subject_idxs.pt')


        test_X = torch.tensor(preprocess_eeg_data(test_X.numpy()), dtype=torch.float32).unsqueeze(1)

        # テストデータセットとデータローダーの作成
        test_dataset = TensorDataset(test_X, test_subject_idxs)
        test_loader = DataLoader(test_dataset, batch_size=cfg.num_batches, shuffle=False)

        # 予測結果を格納するリスト
        predictions = []

        # モデルの定義
        model = ModelClass(num_classes=cfg.num_classes).to(device)

        # 保存されたモデルのファイル名のリスト
        model_files = [f'model_best_fold{fold+1}.pt' for fold in range(cfg.n_splits)]  # 5-Foldの例

        # 各Foldのモデルで予測
        for model_file in model_files:
            # モデルのロード
            model.load_state_dict(torch.load(os.path.join(save_folder_name, model_file), map_location=device))
            model.eval()
            
            fold_predictions = []
            for X, subject_idxs, in test_loader:
                X, subject_idxs = X.to(device), subject_idxs.to(device)
                
                with torch.no_grad():
                    # バッチごとのテストデータの予測
                    pred = model(X, subject_idxs)
                    fold_predictions.append(pred.cpu().numpy())
            
            #各Foldの予測結果を結合
            predictions.append(np.concatenate(fold_predictions, axis=0))

        # 予測結果の平均を計算
        avg_predictions = np.mean(predictions, axis=0)

        # 平均化された予測結果を保存
        mean_acc = 0.05538
        submission_file_path = os.path.join(save_folder_name, f"submission_{mean_acc:.5f}.npy")
        np.save(submission_file_path, avg_predictions)
        cprint(f"Submission {avg_predictions.shape} saved at {submission_file_path}", "cyan")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total run time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    run()
