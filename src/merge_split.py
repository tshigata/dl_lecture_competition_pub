# import os
# import torch
# import numpy as np

# # データを読み込む関数
# def load_data(file_path):
#     return torch.load(file_path)

# # テスト用ダミーデータを生成
# # train_X = np.arange(1, 25).reshape(24, 1, 1)
# # val_X = np.arange(1, 9).reshape(8, 1, 1)
# # train_y = np.arange(1, 25)  # ダミーのラベル
# # val_y = np.arange(25, 33)     # ダミーのラベル
# # train_subject = np.tile(np.arange(4), 6)  # 0, 1, 2, 3が6回繰り返される
# # val_subject = np.tile(np.arange(4), 2)    # 0, 1, 2, 3が2回繰り返される

# # データの読み込み
# train_X = load_data('data/train_X.pt')
# val_X = load_data('data/val_X.pt')
# train_y = load_data('data/train_y.pt')
# val_y = load_data('data/val_y.pt')
# train_subject = load_data('data/train_subject_idxs.pt')
# val_subject = load_data('data/val_subject_idxs.pt')

# # データを統合
# train_val_X = np.concatenate((train_X, val_X), axis=0)
# train_val_y = np.concatenate((train_y, val_y), axis=0)
# train_val_subject = np.concatenate((train_subject, val_subject), axis=0)

# # 各Subject IDごとにフォルダに分ける
# unique_subjects = np.unique(train_val_subject)

# for subject in unique_subjects:
#     subject_indices = np.where(train_val_subject == subject)[0]
    
#     subject_X = train_val_X[subject_indices]
#     subject_y = train_val_y[subject_indices]
    
#     folder_name = f'data{subject}'
    
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)
    
#     torch.save(subject_X, os.path.join(folder_name, 'train_val_X.pt'))
#     torch.save(subject_y, os.path.join(folder_name, 'train_val_y.pt'))
#     torch.save(subject_indices, os.path.join(folder_name, 'subject_idxs.pt'))

# # 各フォルダからデータを読み込んで表示
# for subject in unique_subjects:
#     folder_name = f'data{subject}'
    
#     loaded_X = torch.load(os.path.join(folder_name, 'train_val_X.pt'))
#     loaded_y = torch.load(os.path.join(folder_name, 'train_val_y.pt'))
#     loaded_indices = torch.load(os.path.join(folder_name, 'subject_idxs.pt'))
    
#     print(f"Data for Subject {subject}:")
#     print(f"train_val_X: {loaded_X[:10]}")
#     print(f"train_val_y: {loaded_y[:10]}")
#     print(f"subject_indices: {loaded_indices[:10]}")


# import os
# import torch
# import numpy as np

# # # データを生成（テストデータ）
# # test_X = np.arange(1, 17).reshape(16, 1, 1)
# # test_subject = np.tile(np.arange(4), 4)  # 0, 1, 2, 3が4回繰り返される

# # データの読み込み
# test_X = load_data('data/test_X.pt')
# test_subject = load_data('data/test_subject_idxs.pt')

# # 各Subject IDごとにフォルダに分ける
# unique_subjects = np.unique(test_subject)

# for subject in unique_subjects:
#     subject_indices = np.where(test_subject == subject)[0]
    
#     subject_X = test_X[subject_indices]
    
#     folder_name = f'data{subject}'
    
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)
    
#     torch.save(subject_X, os.path.join(folder_name, 'test_X.pt'))
#     torch.save(subject_indices, os.path.join(folder_name, 'test_subject_idxs.pt'))

# # 各フォルダからデータを読み込んで表示
# for subject in unique_subjects:
#     folder_name = f'data{subject}'
    
#     loaded_X = torch.load(os.path.join(folder_name, 'test_X.pt'))
#     loaded_indices = torch.load(os.path.join(folder_name, 'test_subject_idxs.pt'))
    
#     print(f"Test Data for Subject {subject}:")
#     print(f"test_X: {loaded_X[:10]}")
#     print(f"test_subject_indices: {loaded_indices[:10]}")







import os
import torch
import numpy as np

# データを読み込む関数
def load_data(file_path):
    return torch.load(file_path)

# データの読み込み
train_X = load_data('data/train_X.pt')
val_X = load_data('data/val_X.pt')
train_y = load_data('data/train_y.pt')
val_y = load_data('data/val_y.pt')
train_subject = load_data('data/train_subject_idxs.pt')
val_subject = load_data('data/val_subject_idxs.pt')

# 各Subject IDごとにフォルダに分ける関数
def split_and_save_data(X, y, subject, data_prefix):
    unique_subjects = np.unique(subject)
    
    for subject_id in unique_subjects:
        subject_indices = np.where(subject == subject_id)[0]
        
        subject_X = X[subject_indices]
        subject_y = y[subject_indices]
        
        folder_name = f'data{subject_id}'
        
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        
        torch.save(subject_X, os.path.join(folder_name, f'{data_prefix}_X.pt'))
        torch.save(subject_y, os.path.join(folder_name, f'{data_prefix}_y.pt'))
        torch.save(subject_indices, os.path.join(folder_name, f'{data_prefix}_idxs.pt'))

# トレーニングとバリデーションデータを分割して保存
split_and_save_data(train_X, train_y, train_subject, 'train')
split_and_save_data(val_X, val_y, val_subject, 'val')

# 各フォルダからデータを読み込んで表示（最初の10個の要素）
for subject in np.unique(np.concatenate((train_subject, val_subject))):
    folder_name = f'data{subject}'
    
    train_loaded_X = torch.load(os.path.join(folder_name, 'train_X.pt'))
    train_loaded_y = torch.load(os.path.join(folder_name, 'train_y.pt'))
    train_loaded_indices = torch.load(os.path.join(folder_name, 'train_subject_idxs.pt'))
    
    val_loaded_X = torch.load(os.path.join(folder_name, 'val_X.pt'))
    val_loaded_y = torch.load(os.path.join(folder_name, 'val_y.pt'))
    val_loaded_indices = torch.load(os.path.join(folder_name, 'val_subject_idxs.pt'))
    
    print(f"Data for Subject {subject} - Train:")
    print(f"train_X: {train_loaded_X[:10]}")
    print(f"train_y: {train_loaded_y[:10]}")
    print(f"train_subject_idxs: {train_loaded_indices[:10]}")
    
    print(f"Data for Subject {subject} - Val:")
    print(f"val_X: {val_loaded_X[:10]}")
    print(f"val_y: {val_loaded_y[:10]}")
    print(f"val_subject_idxs: {val_loaded_indices[:10]}")

# テストデータの読み込み
test_X = load_data('data/test_X.pt')
test_subject = load_data('data/test_subject_idxs.pt')

# テストデータを分割して保存
split_and_save_data(test_X, np.zeros(len(test_X)), test_subject, 'test')

# 各フォルダからテストデータを読み込んで表示（最初の10個の要素）
for subject in np.unique(test_subject):
    folder_name = f'data{subject}'
    
    test_loaded_X = torch.load(os.path.join(folder_name, 'test_X.pt'))
    test_loaded_indices = torch.load(os.path.join(folder_name, 'test_subject_idxs.pt'))
    
    print(f"Test Data for Subject {subject}:")
    print(f"test_X: {test_loaded_X[:10]}")
    print(f"test_subject_idxs: {test_loaded_indices[:10]}")
