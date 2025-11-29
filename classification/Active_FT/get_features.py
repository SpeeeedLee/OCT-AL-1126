import sys, os

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from Classification.data import get_data, get_num_train, copy_original_images_by_index
from Classification.model import get_resnet18_classifier, train_model, train_model_w_sam, train_model_w_trustAL, eval_model
from AL_strategy.uncertainty import conf
from AL_strategy.diversity import coreset
from AL_strategy.hybrid import badge

from feature_extract.util import *
from feature_extract.medimage_insight import get_latent_features_med

from model_merging.simple_avg import simple_avg


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, choices=['easy', 'medium', 'hard'], help='')
    parser.add_argument('--device', type=str, default='cuda:0')   

    parser.add_argument('--extractor', type=str, choices=['resnet18_ae', \
                                                          'resnet18_simclr', 'resnet18_pretrained', \
                                                          'resnet18_simclr_sam', \
                                                          'resnet18_simclr_ep5', 'resnet18_simclr_ep20', 'resnet18_simclr_ep40', 'resnet18_simclr_ep60', 'resnet18_simclr_ep80', \
                                                          'resnet18_simclr_dim32', \
                                                          'resnet34_simclr', 'resnet34_pretrained', \
                                                          'resnet50_simclr', 'resnet50_pretrained', \
                                                          'resnet50_pretrained_layer1', 'resnet50_pretrained_layer2', 'resnet50_pretrained_layer3',\
                                                          'resnet50_simclr_layer1', 'resnet50_simclr_layer2', 'resnet50_simclr_layer3', \
                                                          'resnet152_simclr', 'resnet152_pretrained', \
                                                          'moblienetv3_small_pretrained', \
                                                          'dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant', \
                                                          'MedImageInsight',
                                                          'clip_base_32', 'clip_base_16', 'clip_large_14', 'clip_large_14_336' 
                                                          ], help='')

    # dimension reduction related
    parser.add_argument('--dim_reduction', type=str, choices=['pca'], help='')
    parser.add_argument('--dim_portion', type=float, help='')

    # model merging related
    parser.add_argument('--model_merging', action='store_true', help='')
    parser.add_argument('--n_models', type=int, help='number of models merged')
    
    # training methods related
    parser.add_argument('--sam', action='store_true', help='apply shapeness aware minimization')
    
    return parser.parse_args()


def initialize_model(num_classes, pretrained, learning_rate):
    model = get_resnet18_classifier(num_classes = num_classes, pretrained = pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer_ = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler_ = lr_scheduler.StepLR(optimizer_, step_size=5, gamma=0.1)

    return model, criterion, optimizer_, lr_scheduler_


def initialize_simclr_model(num_classes, learning_rate):
    import torch
    from feature_extract.extractor import ResNetSimCLR 
    model = ResNetSimCLR('resnet18', 32)
    state_dict = torch.load('./feature_extract/simclr_resnet.pkl')
    model.load_state_dict(state_dict, strict=False)
    in_features = model.backbone.fc[2].in_features  # 512
    model.backbone.fc = nn.Linear(in_features, num_classes, bias=True) # --> 只有一層，而非mlp，這樣才公平!
    # print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer_ = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler_ = lr_scheduler.StepLR(optimizer_, step_size=5, gamma=0.1)
    return model, criterion, optimizer_, lr_scheduler_


def initialize_ae_model(num_classes, learning_rate):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.optim import lr_scheduler
    from SSL.auto_encoder.classes.resnet_autoencoder import AE

    # 1) 载入 AE 并加载 checkpoint
    ae = AE('default')
    # ckpt = torch.load('./SSL/auto_encoder/autoencoder.ckpt', map_location='cpu')
    ckpt = torch.load('./SSL/auto_encoder/autoencoder_1.ckpt', map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)
    ae.load_state_dict(state_dict, strict=False)

    # 2) 拿到 encoder
    encoder = ae.encoder

    # 3) 在外面用 Sequential 串起来
    model = nn.Sequential(
        encoder,                                # -> [B,512,H,W]
        nn.AdaptiveAvgPool2d((1, 1)),           # -> [B,512,1,1]
        nn.Flatten(),                           # -> [B,512]
        nn.Linear(512, num_classes, bias=True)  # -> [B,num_classes]
    )

    # 4) 损失、优化器、调度器
    criterion    = nn.CrossEntropyLoss()
    optimizer_   = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler_= lr_scheduler.StepLR(optimizer_, step_size=5, gamma=0.1)

    # 打印确认一下
    # print(model)

    return model, criterion, optimizer_, lr_scheduler_


def pca(feature_dict, n_components):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize
    import numpy as np
    
    keys = list(feature_dict.keys())
    X = np.vstack([feature_dict[k] for k in keys])   # shape (N, D)

    # 2) 指定要保留的方差比例，比如 0.95 表示保留 95% 的总方差
    pca = PCA(n_components=n_components, svd_solver='full')
    X_reduced = pca.fit_transform(X)  # shape (N, d) 中的 d 是自动选出的主成分个数
    X_norm = normalize(X_reduced, norm='l2')
    
    # print(f"原维度 = {X.shape[1]}, 降维后 = {X_reduced.shape[1]}")
    # print(f"实际保留方差比例 = {pca.explained_variance_ratio_.sum():.4f}")
    
    print(f"原维度 = {X.shape[1]}, 降维后 = {X_norm.shape[1]}")
    print(f"实际保留方差比例 = {pca.explained_variance_ratio_.sum():.4f}")

    # 3) 把降维后的特征再映射回 dict
    feature_dict_pca = {k: X_reduced[i] for i, k in enumerate(keys)}

    return feature_dict_pca                


def main():
    
    args = parse_arguments()
    if args.task_type == 'easy':
        num_classes = 2
        data_dir = './ds/classification/two_class'
    elif args.task_type == 'medium':
        num_classes = 4
        data_dir = './ds/classification/four_class'
    elif args.task_type == 'hard':
        num_classes = 7
        data_dir = './ds/classification/seven_class'
    else:
        raise ValueError()

    
    ## Get Initial Label/Unlabel Idx
    tot_num_train = get_num_train(data_dir)
    print(f'Total Number of Train: {tot_num_train}')
    unlabeled_idx = list(range(tot_num_train))

    ## Get Embeddings
    if args.extractor in ['dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant'] or\
        args.extractor in ['clip_base_32', 'clip_base_16', 'clip_large_14', 'clip_large_14_336']:
        feature_dict = get_latent_features_vit(data_dir, unlabeled_idx, extractor=args.extractor, device=args.device)
    elif args.extractor in ['MedImageInsight']:
        feature_dict = get_latent_features_med(data_dir, unlabeled_idx, device=args.device)
    else:
        feature_dict = get_latent_features(data_dir, unlabeled_idx, extractor=args.extractor, device=args.device)

    ## Optional PCA
    if args.dim_reduction == 'pca':
        print(f"Applying PCA with portion: {args.dim_portion}")
        
        # 計算要保留的主成分數量
        original_dim = list(feature_dict.values())[0].shape[0]  # 獲取原始特徵維度
        n_components = int(original_dim * args.dim_portion)  # 根據比例計算保留的維度
        
        # 進行 PCA 降維
        feature_dict = pca(feature_dict, n_components)
        print(f"PCA completed: {original_dim} -> {n_components} dimensions")

    ## Save features
    keys = sorted(feature_dict.keys())      
    feats = np.vstack([feature_dict[k] for k in keys])
    idxs  = np.array(keys).reshape(-1, 1)             
    arr   = np.concatenate([feats, idxs], axis=1)    
    print(arr[0]) 
    output_dir = './Classification/Active_FT'
    os.makedirs(output_dir, exist_ok=True)
    
    if args.dim_reduction:
        np.save(os.path.join(output_dir, f'features_{args.extractor}_{args.dim_reduction}{args.dim_portion}.npy'), arr)
    else:
        np.save(os.path.join(output_dir, f'features_{args.extractor}.npy'), arr)
    
    print(f"Features saved to {output_dir}")


if __name__ == "__main__":
    main()