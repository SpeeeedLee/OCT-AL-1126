import sys, os

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import argparse
import random
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from Classification.data import get_data, get_num_train, copy_original_images_by_index
from model import get_resnet18_classifier, train_model, train_model_w_sam, train_model_w_trustAL, eval_model
from AL_strategy.uncertainty import conf
from AL_strategy.diversity import coreset
from AL_strategy.hybrid import badge

from feature_extract.util import get_latent_features, get_latent_features_vit, k_means_centroid, k_means_dense_center, k_means_cluster_margin
from feature_extract.util import *
# from feature_extract.medimage_insight import get_latent_features_med ## this will need different packages?
import time


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, choices=['easy', 'medium', 'hard'], help='')

    parser.add_argument('--portion', type=float) 
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', type=str, default='cuda:0')   
    parser.add_argument('--exp_path', type=str, default='./exp_results')   
    parser.add_argument('--epoch', type=int, default=20)   

    # parser.add_argument('--batch_size', type=int, default=8)  # 自動隨portion決定
    parser.add_argument('--lr', type=float, default=5e-5) 
    parser.add_argument('--weight_decay', type=float)   

    ## pretrained_related --> default to using weights pretrained on ImageNet
    parser.add_argument(
        '--no_use_pretrained',
        dest='use_pretrained',
        action='store_false',
        default=True,
        help='Disable loading of pretrained weights'
    )
    parser.add_argument('--pretrained_weights', type=str, choices=['simclr', 'simclr_sam', 'auto_encoder', 'sparK'])   
    parser.add_argument('--simclr_path', type=str, help='Path to the SimCLR pretrained model weights')
    parser.add_argument('--sparK_path', type=str, help='Path to the SparK pretrained model weights')   
    
    
    # cold start related
    parser.add_argument('--cold_start', action='store_true', help='')
    parser.add_argument('--extractor', type=str, choices=['resnet18_ae', \
                                                          'resnet18_simclr', 'resnet18_pretrained', \
                                                          'resnet18_simclr_sam', 
                                                          'resnet18_simclr_pca0.9', 'resnet18_simclr_pca0.95', \
                                                          'resnet18_simclr_ep5', 'resnet18_simclr_ep20', 'resnet18_simclr_ep40', 'resnet18_simclr_ep60', 'resnet18_simclr_ep80', \
                                                          'resnet18_simclr_dim32', \
                                                          'resnet34_simclr', 'resnet34_pretrained', \
                                                          'resnet50_simclr', 'resnet50_pretrained', \
                                                          'resnet50_pretrained_layer1', 'resnet50_pretrained_layer2', 'resnet50_pretrained_layer3',\
                                                          'resnet50_simclr_layer1', 'resnet50_simclr_layer2', 'resnet50_simclr_layer3', \
                                                          'resnet152_simclr', 'resnet152_pretrained', \
                                                          'moblienetv3_small_pretrained', \
                                                          'dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant', \
                                                          'clip_base_32', 'clip_base_16', 'clip_large_14', 'clip_large_14_336', \
                                                          'MedImageInsight'], help='')
    parser.add_argument('--assign_initial', type=str)
    parser.add_argument('--trial_id', type=int, default=0)

    # data selection from feature embeddings space related
    parser.add_argument('--selection', type=str, choices=['kmeans_centroid', 'kmeans_dense', 'hybrid'], help='')
    
    # dimension reduction related
    parser.add_argument('--dim_reduction', type=str, choices=['pca'], help='')
    parser.add_argument('--dim_portion', type=float, help='')

    
    parser.add_argument(
        '--ask_saving',
        dest='ask_saving',
        action='store_true',
        default=False,
    )
    parser.add_argument('--model_save_path', type=str, default='./model_checkpoints/')

    return parser.parse_args()


def initialize_model(num_classes, pretrained, learning_rate, weight_decay):
    model = get_resnet18_classifier(num_classes = num_classes, pretrained = pretrained)
    criterion = nn.CrossEntropyLoss()
    if weight_decay is not None:
        print(f"Set weight decay to {weight_decay}")
        optimizer_ = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer_ = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler_ = lr_scheduler.StepLR(optimizer_, step_size=5, gamma=0.1)
                
    return model, criterion, optimizer_, lr_scheduler_


def initialize_simclr_model(num_classes, learning_rate, weight_decay, simclr_path):
    import torch
    from feature_extract.extractor import ResNetSimCLR 
    model = ResNetSimCLR('resnet18', 32)
    state_dict = torch.load(simclr_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    in_features = model.backbone.fc[2].in_features  # 512
    model.backbone.fc = nn.Linear(in_features, num_classes, bias=True) # --> 只有一層，而非mlp，這樣才公平!
    # print(model)
    criterion = nn.CrossEntropyLoss()
    if weight_decay is not None:
        print(f"Set weight decay to {weight_decay}")
        optimizer_ = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer_ = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler_ = lr_scheduler.StepLR(optimizer_, step_size=5, gamma=0.1)
    return model, criterion, optimizer_, lr_scheduler_

def initialize_sparK_model(num_classes, learning_rate, weight_decay, sparK_path):
    model = get_resnet18_classifier(num_classes = num_classes, pretrained = False)
    ckpt = torch.load(sparK_path, map_location='cpu')
    state_dict = ckpt["module"]
    # 去掉 "module." prefix
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded SparK checkpoint. Missing keys: {missing}, Unexpected keys: {unexpected}")

    criterion = nn.CrossEntropyLoss()
    if weight_decay is not None:
        print(f"Set weight decay to {weight_decay}")
        optimizer_ = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer_ = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler_ = lr_scheduler.StepLR(optimizer_, step_size=5, gamma=0.1)
    return model, criterion, optimizer_, lr_scheduler_

def initialize_ae_model(num_classes, learning_rate, weight_decay):
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
    if weight_decay is not None:
        print(f"Set weight decay to {weight_decay}")
        optimizer_ = optim.AdamW(model.parameters(), lr=learning_rate,  weight_decay=weight_decay)
    else:
        optimizer_ = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler_= lr_scheduler.StepLR(optimizer_, step_size=5, gamma=0.1)

    return model, criterion, optimizer_, lr_scheduler_


def main():
    
    args = parse_arguments()
    print(args.ask_saving)
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

    ## 自動調整batch size
    if args.portion <= 20:
        args.batch_size = 2
    elif args.portion <= 60:
        args.batch_size = 4
    else:
        args.batch_size = 8
    
    
    if args.cold_start == False:
        file_name = f"random{args.seed}.json"
    else:
        if args.dim_reduction is not None:
            file_name = f"cold_start_{args.extractor}_{args.dim_reduction}_{args.dim_portion}.json"
        else:
            file_name = f"cold_start_{args.extractor}.json"
        if args.assign_initial:
            print(f'use the assigned label idx at {args.assign_initial}')
            file_name = f"./assign_initial/cold_start_assign_initial.json"
    
    if args.selection == 'kmeans_dense':
        file_name = file_name.replace('.json', '_kmeans_dense.json')
    elif args.selection == 'hybrid':
        file_name = file_name.replace('.json', '_hybrid.json')

    base, ext = os.path.splitext(file_name)

    suffix = f"_lr{args.lr}_bs{args.batch_size}"

    # 如果有 weight decay，再在后缀加上
    if args.weight_decay:
        suffix += f"_wd{args.weight_decay}"

    # 重新拼回文件名
    file_name = f"{base}{suffix}{ext}"

    print(f'Exp name: {file_name}')

    ## Get Initial Label/Unlabel Idx
    tot_num_train = get_num_train(data_dir)
    print(f'Total Number of Train: {tot_num_train}')
    label_idx = []
    unlabeled_idx = list(range(tot_num_train))
    print(f'===== Load {args.portion}% Data =====')
    target_num = round(tot_num_train * args.portion / 100)
    num_to_label = target_num - len(label_idx)
        
    ## Train Model
    #### First Iteration only    
    ## Load Data
    if args.assign_initial is None:
        random.seed(args.seed)
        if args.cold_start == False:
            to_label_idx = random.sample(unlabeled_idx, num_to_label)
        else:
            print(f"unsing cold start algorithm to select initial label set!")
            if args.extractor in ['dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant']:
                feature_dict = get_latent_features_vit(data_dir, unlabeled_idx, extractor=args.extractor, device=args.device)
            elif args.extractor in ['MedImageInsight']:
                feature_dict = get_latent_features_med(data_dir, unlabeled_idx, device=args.device)
            else:
                feature_dict = get_latent_features(data_dir, unlabeled_idx, extractor=args.extractor, device=args.device)
            
            if args.selection == 'kmeans_centroid':
                to_label_idx = k_means_centroid(feature_dict, num_to_label, int(time.time()))
            elif args.selection == 'kmeans_dense':
                to_label_idx = k_means_dense_center(feature_dict, num_to_label)
            elif args.selection == 'hybrid':
                to_label_idx = hierarchical_k_means_centroid(feature_dict, 6, num_to_label)
            else:
                raise NotImplementedError()
    else:
        with open(args.assign_initial, 'r') as f:
            assigned_idx = json.load(f)
            to_label_idx = assigned_idx[str(args.portion)][args.extractor][str(args.trial_id)]["id"]

    print("len_label_to_add:", num_to_label)
    print(to_label_idx)
    if len(to_label_idx) != num_to_label:
        raise ValueError()
    # copy_original_images_by_index(data_dir, to_label_idx, save_dir='./temp')
    label_idx.extend(to_label_idx)
    unlabeled_idx = list(set(unlabeled_idx) - set(to_label_idx))
    print("len_label:", len(label_idx), "| len_unlabel:" , len(unlabeled_idx))
    data_loaders, dataset_sizes = get_data(data_dir, label_idx, args.batch_size)
    print(dataset_sizes)
    
    ## Train Model 
    print(f'===== Train Model with {args.portion}% data=====')
    ## Initialize Model
    if args.use_pretrained:
        if args.pretrained_weights == 'simclr':
            print('initialize resnet18 using simclr weights!')
            model, criterion, optimizer_, lr_scheduler_ = initialize_simclr_model(num_classes, args.lr, args.weight_decay, args.simclr_path)
        elif args.pretrained_weights == 'sparK':
            print('initialize resnet18 using sparK weights!')
            model, criterion, optimizer_, lr_scheduler_ = initialize_sparK_model(num_classes, args.lr, args.weight_decay, args.sparK_path)
        elif args.pretrained_weights == 'auto_encoder':
            model, criterion, optimizer_, lr_scheduler_ = initialize_ae_model(num_classes, args.lr, args.weight_decay)
        else:
            print('initialize resnet18 using pretrained weights (i.e. ImageNet1K)!')
            model, criterion, optimizer_, lr_scheduler_ = initialize_model(num_classes, True, args.lr, args.weight_decay)
    else:
        print('initialize resnet18 with no pretrained weights!')
        model, criterion, optimizer_, lr_scheduler_ = initialize_model(num_classes, False, args.lr, args.weight_decay)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_trainable_params}")

    _, final_acc = train_model(model, args.device, data_loaders, dataset_sizes, criterion, optimizer_, lr_scheduler_, num_epochs=args.epoch)
    print("Final Acc: ", final_acc)
    
    ## Optional Save Model
    if args.ask_saving:
        user_input = input('Do you want to save the model checkpoint? (y/n): ').strip().lower()
        if user_input in ['y', 'yes']:
            
            # 讓使用者輸入模型路徑
            model_path = input('Please enter the path to save the model: ').strip()
            
            # 基本檢查
            if not model_path:
                print('No path provided. Using default: ./model_checkpoint.pth')
                model_path = './model_checkpoint.pth'
            
            # 確保目錄存在
            model_dir = os.path.dirname(model_path)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            
            # 保存模型
            try:
                torch.save(model.state_dict(), model_path)            
                print(f'Model checkpoint saved to: {model_path}')
            except Exception as e:
                print(f'Error saving model: {e}')
            
        else:
            print('Model checkpoint not saved.')
    ## Save Results
    if args.assign_initial:
        with open(args.assign_initial, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data = {str(float(k)): v for k, v in data.items()}
            if str(args.batch_size) not in data[str(args.portion)][args.extractor][str(args.trial_id)]["perf"]:
                data[str(args.portion)][args.extractor][str(args.trial_id)]["perf"][str(args.batch_size)] = []
            data[str(args.portion)][args.extractor][str(args.trial_id)]["perf"][str(args.batch_size)].append(final_acc)
            f.seek(0)
            json.dump(data, f, indent=4)
            f.truncate()
        print(f'result saved to {args.assign_initial}!')
        return
    if args.use_pretrained:
        if args.pretrained_weights is not None:
            save_path = os.path.join(args.exp_path, f"classification_{args.task_type}", f"cold_start_pretrained_{args.pretrained_weights}")
        else:
            save_path = os.path.join(args.exp_path, f"classification_{args.task_type}", "cold_start")
    else:
        save_path = os.path.join(args.exp_path, f"classification_{args.task_type}", "cold_start_no_pretrained")
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, file_name)
    if not os.path.isfile(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("{}")
    with open(file_path, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data = {str(float(k)): v for k, v in data.items()}
        if str(float(args.portion)) not in data:
            data[str(float(args.portion))] = [final_acc]
        else:
            data[str(float(args.portion))].append(final_acc)
        sorted_data = {k: data[k] for k in sorted(data, key=lambda x: float(x))}
        f.seek(0)
        f.truncate()
        json.dump(sorted_data, f, indent=4, ensure_ascii=False)
    print(f'result saved to {file_path}!')

    return

if __name__ == "__main__":
    main()