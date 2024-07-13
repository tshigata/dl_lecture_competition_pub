import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import torchvision.models as models
from termcolor import cprint

# EEGNetのモデル定義
class EEGNet(nn.Module):
    def __init__(self, num_classes, Chans=271, Samples=128, dropout_rate=0.25):
        super(EEGNet, self).__init__()
        cprint(f"dropout_rate = {dropout_rate}", "light_blue")
        
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

    def forward(self, x, subject_idxs=None):
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

    def forward(self, x, subject_idxs=None):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv1(x)
        x = self.separableConv2(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x

class EEGNetWithSubject(nn.Module):
    def __init__(self, num_classes, Chans=271, Samples=128, dropout_rate=0.5, num_subjects=4):
        super(EEGNetWithSubject, self).__init__()
        
        self.subject_embedding = nn.Embedding(num_subjects, 16)  # 被験者IDのエンベッディング層

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((1, 2))
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(32, 64, (Chans, 1), stride=(1, 1), groups=32, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv1 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv2 = nn.Sequential(
            nn.Conv2d(128, 256, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.flattened_size = 256 * ((Samples // 2 // 2 // 2 // 2))
        self.classify = nn.Sequential(
            nn.Linear(self.flattened_size + 16, 512),  # 埋め込みベクトルのサイズを追加
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, subject_idxs):
        subject_embeds = self.subject_embedding(subject_idxs)
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv1(x)
        x = self.separableConv2(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, subject_embeds), dim=1)
        x = self.classify(x)
        return x

   
class SubjectBatchNorm(nn.Module):
    def __init__(self, num_features, num_subjects):
        super(SubjectBatchNorm, self).__init__()
        self.num_subjects = num_subjects
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features) for _ in range(num_subjects)])

    def forward(self, x, subject_idx):
        out = torch.zeros_like(x)
        for i in range(self.num_subjects):
            mask = (subject_idx == i).unsqueeze(1).unsqueeze(2).unsqueeze(3).float()
            out += self.bns[i](x) * mask
        return out

class EEGNetWithSubjectBatchNorm(nn.Module):
    def __init__(self, num_classes, Chans=271, Samples=128, dropout_rate=0.5, num_subjects=4):
        super(EEGNetWithSubjectBatchNorm, self).__init__()
        
        self.subject_embedding = nn.Embedding(num_subjects, 16)

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            SubjectBatchNorm(32, num_subjects),
            nn.ELU(),
            nn.MaxPool2d((1, 2))
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(32, 64, (Chans, 1), stride=(1, 1), groups=32, bias=False),
            SubjectBatchNorm(64, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv1 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNorm(128, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv2 = nn.Sequential(
            nn.Conv2d(128, 256, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNorm(256, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.flattened_size = 256 * ((Samples // 2 // 2 // 2 // 2))
        self.classify = nn.Sequential(
            nn.Linear(self.flattened_size + 16, 512),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, subject_idxs):
        subject_embeds = self.subject_embedding(subject_idxs)
        x = self.firstconv[0](x)
        x = self.firstconv[1](x, subject_idxs)
        x = self.firstconv[2](x)
        x = self.firstconv[3](x)
        
        x = self.depthwiseConv[0](x)
        x = self.depthwiseConv[1](x, subject_idxs)
        x = self.depthwiseConv[2](x)
        x = self.depthwiseConv[3](x)
        x = self.depthwiseConv[4](x)
        
        x = self.separableConv1[0](x)
        x = self.separableConv1[1](x, subject_idxs)
        x = self.separableConv1[2](x)
        x = self.separableConv1[3](x)
        x = self.separableConv1[4](x)
        
        x = self.separableConv2[0](x)
        x = self.separableConv2[1](x, subject_idxs)
        x = self.separableConv2[2](x)
        x = self.separableConv2[3](x)
        x = self.separableConv2[4](x)
        
        x = x.view(x.size(0), -1)
        x = torch.cat((x, subject_embeds), dim=1)
        x = self.classify(x)
        return x




class EGNetWithSubjectBatchNormAdd(nn.Module):
    def __init__(self, num_classes, Chans=271, Samples=128, dropout_rate=0.5, num_subjects=4):
        super(EGNetWithSubjectBatchNormAdd, self).__init__()
        
        self.subject_embedding = nn.Embedding(num_subjects, 16)

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            SubjectBatchNorm(32, num_subjects),
            nn.ELU(),
            nn.MaxPool2d((1, 2))
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(32, 64, (Chans, 1), stride=(1, 1), groups=32, bias=False),
            SubjectBatchNorm(64, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv1 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNorm(128, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv2 = nn.Sequential(
            nn.Conv2d(128, 256, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNorm(256, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.additional_conv = nn.Sequential(
            nn.Conv2d(256, 512, (1, 10), stride=(1, 1), padding=(0, 5), bias=False),  # 追加の畳み込み層
            SubjectBatchNorm(512, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.flattened_size = 512 * ((Samples // 2 // 2 // 2 // 2 // 2))
        self.classify = nn.Sequential(
            nn.Linear(self.flattened_size + 16, 512),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, subject_idxs):
        subject_embeds = self.subject_embedding(subject_idxs)

        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv1(x)
        x = self.separableConv2(x)
        x = self.additional_conv(x)  # 追加の畳み込み層を適用

        x = x.view(x.size(0), -1)
        x = torch.cat((x, subject_embeds), dim=1)
        x = self.classify(x)
        return x


class EEGNetWithSubjectBatchNormAll(nn.Module):
    def __init__(self, num_classes, Chans=271, Samples=128, dropout_rate=0.5, num_subjects=4):
        super(EEGNetWithSubjectBatchNormAll, self).__init__()
        
        self.subject_embedding = nn.Embedding(num_subjects, 16)

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            SubjectBatchNorm(32, num_subjects),
            nn.ELU(),
            nn.MaxPool2d((1, 2))
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(32, 64, (Chans, 1), stride=(1, 1), groups=32, bias=False),
            SubjectBatchNorm(64, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv1 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNorm(128, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv2 = nn.Sequential(
            nn.Conv2d(128, 256, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNorm(256, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.flattened_size = 256 * ((Samples // 2 // 2 // 2 // 2))
        self.classify = nn.Sequential(
            nn.Linear(self.flattened_size + 16, 1024),  # 隠れ層の次元を増加
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, subject_idxs):
        subject_embeds = self.subject_embedding(subject_idxs)

        x = self.firstconv[0](x)
        x = self.firstconv[1](x, subject_idxs)
        x = self.firstconv[2](x)
        x = self.firstconv[3](x)
        
        x = self.depthwiseConv[0](x)
        x = self.depthwiseConv[1](x, subject_idxs)
        x = self.depthwiseConv[2](x)
        x = self.depthwiseConv[3](x)
        x = self.depthwiseConv[4](x)
        
        x = self.separableConv1[0](x)
        x = self.separableConv1[1](x, subject_idxs)
        x = self.separableConv1[2](x)
        x = self.separableConv1[3](x)
        x = self.separableConv1[4](x)
        
        x = self.separableConv2[0](x)
        x = self.separableConv2[1](x, subject_idxs)
        x = self.separableConv2[2](x)
        x = self.separableConv2[3](x)
        x = self.separableConv2[4](x)
        
        x = x.view(x.size(0), -1)
        x = torch.cat((x, subject_embeds), dim=1)
        x = self.classify(x)
        return x
    

class EEGNetWithSubjectBatchNormAll3(nn.Module):
    def __init__(self, num_classes, Chans=271, Samples=128, dropout_rate=0.5, num_subjects=4):
        super(EEGNetWithSubjectBatchNormAll3, self).__init__()
        print(f"Samples: {Samples}", "Chans: {Chans}")
        self.subject_embedding = nn.Embedding(num_subjects, 16)

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            SubjectBatchNorm(32, num_subjects),
            nn.ELU(),
            nn.MaxPool2d((1, 2))
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(32, 64, (Chans, 1), stride=(1, 1), groups=32, bias=False),
            SubjectBatchNorm(64, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv1 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNorm(128, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv2 = nn.Sequential(
            nn.Conv2d(128, 256, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNorm(256, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )
        self.separableConv3 = nn.Sequential(
            nn.Conv2d(256, 512, (1, 7), stride=(1, 1), padding=(0, 3), bias=False),
            SubjectBatchNorm(512, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.flattened_size = 256 * ((Samples // 2 // 2 // 2))
        print(f"self.flattened_size: {self.flattened_size}")
        self.classify = nn.Sequential(
            nn.Linear(self.flattened_size + 16, 1024),  # 隠れ層の次元を増加
            # nn.Linear(3088, 1024),  # 隠れ層の次元を増加
            # nn.Linear(4112, 1024),  # 隠れ層の次元を増加
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, subject_idxs):
        subject_embeds = self.subject_embedding(subject_idxs)

        x = self.firstconv[0](x)
        x = self.firstconv[1](x, subject_idxs) # batchnormに被験者IDを渡す
        x = self.firstconv[2](x)
        x = self.firstconv[3](x)
        
        x = self.depthwiseConv[0](x)
        x = self.depthwiseConv[1](x, subject_idxs)
        x = self.depthwiseConv[2](x)
        x = self.depthwiseConv[3](x)
        x = self.depthwiseConv[4](x)
        
        x = self.separableConv1[0](x)
        x = self.separableConv1[1](x, subject_idxs)
        x = self.separableConv1[2](x)
        x = self.separableConv1[3](x)
        x = self.separableConv1[4](x)
        
        x = self.separableConv2[0](x)
        x = self.separableConv2[1](x, subject_idxs)
        x = self.separableConv2[2](x)
        x = self.separableConv2[3](x)
        x = self.separableConv2[4](x)

        x = self.separableConv3[0](x)
        x = self.separableConv3[1](x, subject_idxs)
        x = self.separableConv3[2](x)
        x = self.separableConv3[3](x)
        x = self.separableConv3[4](x)
        
        # print(f"x.shape1: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"x.shape2: {x.shape}")
        x = torch.cat((x, subject_embeds), dim=1)
        # print(f"x.shape3: {x.shape}")
        x = self.classify(x)
        # print(f"x.shape4: {x.shape}")
        return x
    
class EEGNetWithSubjectBatchNormAll3SubjectInjection(nn.Module):
    def __init__(self, num_classes, Chans=271, Samples=128, dropout_rate=0.5, num_subjects=4):
        super(EEGNetWithSubjectBatchNormAll3SubjectInjection, self).__init__()
        print(f"Samples: {Samples}", f"Chans: {Chans}")
        self.subject_embedding = nn.Embedding(num_subjects, 16)

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            SubjectBatchNorm(32, num_subjects),
            nn.ELU(),
            nn.MaxPool2d((1, 2))
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(32 + 16, 48, (Chans, 1), stride=(1, 1), groups=48, bias=False),
            SubjectBatchNorm(48, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv1 = nn.Sequential(
            nn.Conv2d(48 + 16, 128, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNorm(128, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv2 = nn.Sequential(
            nn.Conv2d(128 + 16, 256, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNorm(256, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )
        self.separableConv3 = nn.Sequential(
            nn.Conv2d(256 + 16, 512, (1, 7), stride=(1, 1), padding=(0, 3), bias=False),
            SubjectBatchNorm(512, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        # Calculate the flattened size dynamically
        def conv2d_output_size(size, kernel_size, stride=1, padding=0, dilation=1):
            return (size + 2*padding - dilation*(kernel_size - 1) - 1) // stride + 1
        
        def pool_output_size(size, kernel_size, stride=1, padding=0, dilation=1):
            return (size - kernel_size) // stride + 1

        # First conv and pool layers
        size = Samples
        size = conv2d_output_size(size, 51, padding=25)
        size = pool_output_size(size, 2, stride=2)
        
        # Depthwise conv and pool layers
        size = pool_output_size(size, 2, stride=2)
        
        # Separable conv1 and pool layers
        size = conv2d_output_size(size, 15, padding=7)
        size = pool_output_size(size, 2, stride=2)
        
        # Separable conv2 and pool layers
        size = conv2d_output_size(size, 15, padding=7)
        size = pool_output_size(size, 2, stride=2)
        
        # Separable conv3 and pool layers
        size = conv2d_output_size(size, 7, padding=3)
        size = pool_output_size(size, 2, stride=2)

        self.flattened_size = 512 * size
        print(f"self.flattened_size: {self.flattened_size}")

        self.classify = nn.Sequential(
            # nn.Linear(self.flattened_size + 16, 1024),  # 隠れ層の次元を増加
            nn.Linear(4112, 1024),  # 隠れ層の次元を増加
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, subject_idxs):
        subject_embeds = self.subject_embedding(subject_idxs)

        x = self.firstconv[0](x)
        x = self.firstconv[1](x, subject_idxs)  # batchnormに被験者IDを渡す
        x = self.firstconv[2](x)
        x = self.firstconv[3](x)
        
        # Add subject_embeds to x before passing through depthwiseConv
        x = torch.cat((x, subject_embeds.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))), dim=1)
        
        x = self.depthwiseConv[0](x)
        x = self.depthwiseConv[1](x, subject_idxs)
        x = self.depthwiseConv[2](x)
        x = self.depthwiseConv[3](x)
        x = self.depthwiseConv[4](x)
        
        # Add subject_embeds to x before passing through separableConv1
        x = torch.cat((x, subject_embeds.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))), dim=1)
        
        x = self.separableConv1[0](x)
        x = self.separableConv1[1](x, subject_idxs)
        x = self.separableConv1[2](x)
        x = self.separableConv1[3](x)
        x = self.separableConv1[4](x)
        
        # Add subject_embeds to x before passing through separableConv2
        x = torch.cat((x, subject_embeds.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))), dim=1)
        
        x = self.separableConv2[0](x)
        x = self.separableConv2[1](x, subject_idxs)
        x = self.separableConv2[2](x)
        x = self.separableConv2[3](x)
        x = self.separableConv2[4](x)

        # Add subject_embeds to x before passing through separableConv3
        x = torch.cat((x, subject_embeds.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))), dim=1)

        x = self.separableConv3[0](x)
        x = self.separableConv3[1](x, subject_idxs)
        x = self.separableConv3[2](x)
        x = self.separableConv3[3](x)
        x = self.separableConv3[4](x)
        
        x = x.view(x.size(0), -1)
        x = torch.cat((x, subject_embeds), dim=1)
        x = self.classify(x)
        return x



import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

class EEGNetGRU(nn.Module):
    def __init__(self, num_classes, Chans=271, Samples=128, dropout_rate=0.5, num_subjects=4):
        super(EEGNetGRU, self).__init__()
        
        self.subject_embedding = nn.Embedding(num_subjects, 16)

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            SubjectBatchNorm(32, num_subjects),
            nn.ELU(),
            nn.MaxPool2d((1, 2))
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(32, 64, (Chans, 1), stride=(1, 1), groups=32, bias=False),
            SubjectBatchNorm(64, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv1 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNorm(128, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv2 = nn.Sequential(
            nn.Conv2d(128, 256, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNorm(256, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )
        self.separableConv3 = nn.Sequential(
            nn.Conv2d(256, 512, (1, 7), stride=(1, 1), padding=(0, 3), bias=False),
            SubjectBatchNorm(512, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.gru_input_size = 512
        self.flattened_size = Samples // 2 // 2 // 2 // 2
        
        self.gru = nn.GRU(input_size=self.gru_input_size, hidden_size=256, num_layers=2, batch_first=True, dropout=dropout_rate)
        
        self.classify = nn.Sequential(
            nn.Linear(256 + 16, 1024),  # GRUの出力に被験者埋め込みを結合
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, subject_idxs):
        subject_embeds = self.subject_embedding(subject_idxs)

        x = self.firstconv[0](x)
        x = self.firstconv[1](x, subject_idxs)
        x = self.firstconv[2](x)
        x = self.firstconv[3](x)
        
        x = self.depthwiseConv[0](x)
        x = self.depthwiseConv[1](x, subject_idxs)
        x = self.depthwiseConv[2](x)
        x = self.depthwiseConv[3](x)
        x = self.depthwiseConv[4](x)
        
        x = self.separableConv1[0](x)
        x = self.separableConv1[1](x, subject_idxs)
        x = self.separableConv1[2](x)
        x = self.separableConv1[3](x)
        x = self.separableConv1[4](x)
        
        x = self.separableConv2[0](x)
        x = self.separableConv2[1](x, subject_idxs)
        x = self.separableConv2[2](x)
        x = self.separableConv2[3](x)
        x = self.separableConv2[4](x)

        x = self.separableConv3[0](x)
        x = self.separableConv3[1](x, subject_idxs)
        x = self.separableConv3[2](x)
        x = self.separableConv3[3](x)
        x = self.separableConv3[4](x)
        
        # Flatten the output for GRU
        x = x.view(x.size(0), x.size(3), -1)
        
        # GRU
        x, _ = self.gru(x)
        x = x[:, -1, :]  # 最後の時間ステップの出力を使用
        
        x = torch.cat((x, subject_embeds), dim=1)
        x = self.classify(x)
        return x



class EEGNetWithSubjectBatchNormAll_Org(nn.Module):
    def __init__(self, num_classes, Chans=271, Samples=281, dropout_rate=0.5, num_subjects=4):
        super(EEGNetWithSubjectBatchNormAll_Org, self).__init__()
        print(f"Samples: {Samples}", "Chans: {Chans}")
        self.subject_embedding = nn.Embedding(num_subjects, 16)

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            SubjectBatchNorm(32, num_subjects),
            nn.ELU(),
            nn.MaxPool2d((1, 2))
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(32, 64, (Chans, 1), stride=(1, 1), groups=32, bias=False),
            SubjectBatchNorm(64, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv1 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNorm(128, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv2 = nn.Sequential(
            nn.Conv2d(128, 256, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNorm(256, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )
        self.separableConv3 = nn.Sequential(
            nn.Conv2d(256, 512, (1, 7), stride=(1, 1), padding=(0, 3), bias=False),
            SubjectBatchNorm(512, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.flattened_size = 256 * ((Samples // 2 // 2 // 2 // 2))
        # print(f"self.flattened_size: {self.flattened_size}")
        self.classify = nn.Sequential(
            nn.Linear(self.flattened_size + 16, 1024),  # 隠れ層の次元を増加
            # nn.Linear(3088, 1024),  # 隠れ層の次元を増加
            # nn.Linear(4112, 1024),  # 隠れ層の次元を増加
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, subject_idxs):
        subject_embeds = self.subject_embedding(subject_idxs)

        x = self.firstconv[0](x)
        x = self.firstconv[1](x, subject_idxs) # batchnormに被験者IDを渡す
        x = self.firstconv[2](x)
        x = self.firstconv[3](x)
        
        x = self.depthwiseConv[0](x)
        x = self.depthwiseConv[1](x, subject_idxs)
        x = self.depthwiseConv[2](x)
        x = self.depthwiseConv[3](x)
        x = self.depthwiseConv[4](x)
        
        x = self.separableConv1[0](x)
        x = self.separableConv1[1](x, subject_idxs)
        x = self.separableConv1[2](x)
        x = self.separableConv1[3](x)
        x = self.separableConv1[4](x)
        
        x = self.separableConv2[0](x)
        x = self.separableConv2[1](x, subject_idxs)
        x = self.separableConv2[2](x)
        x = self.separableConv2[3](x)
        x = self.separableConv2[4](x)

        x = self.separableConv3[0](x)
        x = self.separableConv3[1](x, subject_idxs)
        x = self.separableConv3[2](x)
        x = self.separableConv3[3](x)
        x = self.separableConv3[4](x)
        
        # print(f"x.shape1: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"x.shape2: {x.shape}")
        x = torch.cat((x, subject_embeds), dim=1)
        # print(f"x.shape3: {x.shape}")
        x = self.classify(x)
        # print(f"x.shape4: {x.shape}")
        return x


# SubjectBatchNormクラスの定義
class SubjectBatchNormA(nn.Module):
    def __init__(self, num_features, num_subjects):
        super(SubjectBatchNormA, self).__init__()
        self.num_features = num_features
        self.num_subjects = num_subjects
        self.bn_layers = nn.ModuleList([nn.BatchNorm2d(num_features) for _ in range(num_subjects)])

    def forward(self, x, subject_idxs):
        out = torch.zeros_like(x)
        for i in range(self.num_subjects):
            mask = (subject_idxs == i).view(-1, 1, 1, 1).expand_as(x)
            if torch.any(mask):
                # out[mask] = self.bn_layers[i](x[mask])
                out += self.bn_layers[i](x * mask)
        return out

# ImprovedEEGNetクラスの定義
class ImprovedEEGNet(nn.Module):
    def __init__(self, num_classes, Chans=271, Samples=128, dropout_rate=0.5, num_subjects=4):
        super(ImprovedEEGNet, self).__init__()

        self.subject_embedding = nn.Embedding(num_subjects, 16)

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            SubjectBatchNormA(32, num_subjects),
            nn.ELU(),
            nn.MaxPool2d((1, 2))
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(32, 64, (Chans, 1), stride=(1, 1), groups=32, bias=False),
            SubjectBatchNormA(64, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.residual_block1 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 1), stride=(1, 1), bias=False),
            SubjectBatchNormA(64, num_subjects),
            nn.ELU()
        )

        self.separableConv1 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNormA(128, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv2 = nn.Sequential(
            nn.Conv2d(128, 256, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNormA(256, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv3 = nn.Sequential(
            nn.Conv2d(256, 512, (1, 7), stride=(1, 1), padding=(0, 3), bias=False),
            SubjectBatchNormA(512, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.attention = nn.Sequential(
            # nn.Conv2d(512, 64, (1, 1), bias=False),
            nn.Conv2d(512, 512, (1, 1), bias=False),
            nn.Sigmoid()
        )

        self.flattened_size = 512 * ((Samples // 2 // 2 // 2 // 2 // 2) * 1)
        print(self.flattened_size)
        self.classify = nn.Sequential(
            nn.Linear(self.flattened_size + 16, 1024),
            # nn.Linear(self.get_flattened_size(Chans, Samples) + 16, 1024),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, subject_idxs):
        subject_embeds = self.subject_embedding(subject_idxs)

        x = self.firstconv[0](x)
        x = self.firstconv[1](x, subject_idxs)
        x = self.firstconv[2](x)
        x = self.firstconv[3](x)

        x = self.depthwiseConv[0](x)
        x = self.depthwiseConv[1](x, subject_idxs)
        x = self.depthwiseConv[2](x)
        x = self.depthwiseConv[3](x)
        x = self.depthwiseConv[4](x)

        residual = x
        x = self.residual_block1[0](x)
        x = self.residual_block1[1](x, subject_idxs)
        x = self.residual_block1[2](x)
        x = x + residual

        x = self.separableConv1[0](x)
        x = self.separableConv1[1](x, subject_idxs)
        x = self.separableConv1[2](x)
        x = self.separableConv1[3](x)
        x = self.separableConv1[4](x)

        x = self.separableConv2[0](x)
        x = self.separableConv2[1](x, subject_idxs)
        x = self.separableConv2[2](x)
        x = self.separableConv2[3](x)
        x = self.separableConv2[4](x)

        x = self.separableConv3[0](x)
        x = self.separableConv3[1](x, subject_idxs)
        x = self.separableConv3[2](x)
        x = self.separableConv3[3](x)
        x = self.separableConv3[4](x)

        attention_weights = self.attention(x)
        x = x * attention_weights

        x = x.view(x.size(0), -1)
        x = torch.cat((x, subject_embeds), dim=1)
        x = self.classify(x)
        return x
    

    # ImprovedEEGNetクラスの定義(畳み込み層を減らした)
class ImprovedEEGNet2(nn.Module):
    def __init__(self, num_classes, Chans=271, Samples=128, dropout_rate=0.5, num_subjects=4):
        super(ImprovedEEGNet2, self).__init__()

        self.subject_embedding = nn.Embedding(num_subjects, 16)

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            SubjectBatchNormA(32, num_subjects),
            nn.ELU(),
            nn.MaxPool2d((1, 2))
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(32, 64, (Chans, 1), stride=(1, 1), groups=32, bias=False),
            SubjectBatchNormA(64, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.residual_block1 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 1), stride=(1, 1), bias=False),
            SubjectBatchNormA(64, num_subjects),
            nn.ELU()
        )

        self.separableConv1 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNormA(128, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv2 = nn.Sequential(
            nn.Conv2d(128, 256, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNormA(256, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        # self.separableConv3 = nn.Sequential(
        #     nn.Conv2d(256, 512, (1, 7), stride=(1, 1), padding=(0, 3), bias=False),
        #     SubjectBatchNormA(512, num_subjects),
        #     nn.ELU(),
        #     nn.AvgPool2d((1, 2), stride=(1, 2)),
        #     nn.Dropout(dropout_rate)
        # )

        self.attention = nn.Sequential(
            # nn.Conv2d(512, 64, (1, 1), bias=False),
            # nn.Conv2d(512, 512, (1, 1), bias=False),
            nn.Conv2d(256, 256, (1, 1), bias=False),
            nn.Sigmoid()
        )

        self.flattened_size = 256 * ((Samples // 2 // 2 // 2 // 2) * 1)
        # print(self.flattened_size+16)
        self.classify = nn.Sequential(
            nn.Linear(self.flattened_size + 16, 1024),
            # nn.Linear(self.get_flattened_size(Chans, Samples) + 16, 1024),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, subject_idxs):
        subject_embeds = self.subject_embedding(subject_idxs)

        x = self.firstconv[0](x)
        x = self.firstconv[1](x, subject_idxs)
        x = self.firstconv[2](x)
        x = self.firstconv[3](x)

        x = self.depthwiseConv[0](x)
        x = self.depthwiseConv[1](x, subject_idxs)
        x = self.depthwiseConv[2](x)
        x = self.depthwiseConv[3](x)
        x = self.depthwiseConv[4](x)

        residual = x
        x = self.residual_block1[0](x)
        x = self.residual_block1[1](x, subject_idxs)
        x = self.residual_block1[2](x)
        x = x + residual

        x = self.separableConv1[0](x)
        x = self.separableConv1[1](x, subject_idxs)
        x = self.separableConv1[2](x)
        x = self.separableConv1[3](x)
        x = self.separableConv1[4](x)

        x = self.separableConv2[0](x)
        x = self.separableConv2[1](x, subject_idxs)
        x = self.separableConv2[2](x)
        x = self.separableConv2[3](x)
        x = self.separableConv2[4](x)

        # x = self.separableConv3[0](x)
        # x = self.separableConv3[1](x, subject_idxs)
        # x = self.separableConv3[2](x)
        # x = self.separableConv3[3](x)
        # x = self.separableConv3[4](x)

        attention_weights = self.attention(x)
        x = x * attention_weights

        x = x.view(x.size(0), -1)
        x = torch.cat((x, subject_embeds), dim=1)
        # print(x.shape)
        x = self.classify(x)
        return x


    # ImprovedEEGNetクラスの定義(畳み込み層を減らし,R抜き)
class ImprovedEEGNet3(nn.Module):
    def __init__(self, num_classes, Chans=271, Samples=128, dropout_rate=0.5, num_subjects=4):
        super(ImprovedEEGNet3, self).__init__()

        self.subject_embedding = nn.Embedding(num_subjects, 16)

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 32, (1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            SubjectBatchNormA(32, num_subjects),
            nn.ELU(),
            nn.MaxPool2d((1, 2))
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(32, 64, (Chans, 1), stride=(1, 1), groups=32, bias=False),
            SubjectBatchNormA(64, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.residual_block1 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 1), stride=(1, 1), bias=False),
            SubjectBatchNormA(64, num_subjects),
            nn.ELU()
        )

        self.separableConv1 = nn.Sequential(
            nn.Conv2d(64, 128, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNormA(128, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        self.separableConv2 = nn.Sequential(
            nn.Conv2d(128, 256, (1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            SubjectBatchNormA(256, num_subjects),
            nn.ELU(),
            nn.AvgPool2d((1, 2), stride=(1, 2)),
            nn.Dropout(dropout_rate)
        )

        # self.separableConv3 = nn.Sequential(
        #     nn.Conv2d(256, 512, (1, 7), stride=(1, 1), padding=(0, 3), bias=False),
        #     SubjectBatchNormA(512, num_subjects),
        #     nn.ELU(),
        #     nn.AvgPool2d((1, 2), stride=(1, 2)),
        #     nn.Dropout(dropout_rate)
        # )

        self.attention = nn.Sequential(
            # nn.Conv2d(512, 64, (1, 1), bias=False),
            # nn.Conv2d(512, 512, (1, 1), bias=False),
            nn.Conv2d(256, 256, (1, 1), bias=False),
            nn.Sigmoid()
        )

        self.flattened_size = 256 * ((Samples // 2 // 2 // 2 // 2) * 1)
        # print(self.flattened_size+16)
        self.classify = nn.Sequential(
            nn.Linear(self.flattened_size + 16, 1024),
            # nn.Linear(self.get_flattened_size(Chans, Samples) + 16, 1024),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, subject_idxs):
        subject_embeds = self.subject_embedding(subject_idxs)

        x = self.firstconv[0](x)
        x = self.firstconv[1](x, subject_idxs)
        x = self.firstconv[2](x)
        x = self.firstconv[3](x)

        x = self.depthwiseConv[0](x)
        x = self.depthwiseConv[1](x, subject_idxs)
        x = self.depthwiseConv[2](x)
        x = self.depthwiseConv[3](x)
        x = self.depthwiseConv[4](x)

        # residual = x
        x = self.residual_block1[0](x)
        x = self.residual_block1[1](x, subject_idxs)
        x = self.residual_block1[2](x)
        # x = x + residual

        x = self.separableConv1[0](x)
        x = self.separableConv1[1](x, subject_idxs)
        x = self.separableConv1[2](x)
        x = self.separableConv1[3](x)
        x = self.separableConv1[4](x)

        x = self.separableConv2[0](x)
        x = self.separableConv2[1](x, subject_idxs)
        x = self.separableConv2[2](x)
        x = self.separableConv2[3](x)
        x = self.separableConv2[4](x)

        # x = self.separableConv3[0](x)
        # x = self.separableConv3[1](x, subject_idxs)
        # x = self.separableConv3[2](x)
        # x = self.separableConv3[3](x)
        # x = self.separableConv3[4](x)

        attention_weights = self.attention(x)
        x = x * attention_weights

        x = x.view(x.size(0), -1)
        x = torch.cat((x, subject_embeds), dim=1)
        # print(x.shape)
        x = self.classify(x)
        return x