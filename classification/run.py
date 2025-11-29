import sys, os

print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import argparse
import random
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from classification.data import get_data, get_num_train
from model import get_resnet18_classifier, train_model, train_model_w_sam

from feature_extract.util import ResNetSimCLR 

from AL_strategy.uncertainty import conf, entropy, margin
from AL_strategy.diversity import coreset, compute_density_scores
from AL_strategy.hybrid import badge





def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, choices=['easy', 'medium', 'hard'], help='')
    parser.add_argument('--AL_strategy', type=str, choices=['random', 'conf', 'entropy', 'margin', 'coreset', 'badge'], help='')

    # AL related setup
    parser.add_argument('--portion_start', type=float, default=10)
    parser.add_argument('--portion_end', type=float, default=110)
    parser.add_argument('--portion_interval', type=float, default=10)   
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', type=str, default='cuda:0')   
    parser.add_argument('--exp_path', type=str, default='./exp_results')   
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--test_id', type=int)   


    # pretrained_weights
    parser.add_argument('--pretrained_weights', type=str, choices=['simclr', 'auto_encoder'])   
    parser.add_argument('--simclr_path', type=str, help='Path to the SimCLR pretrained model weights')
    
    # training hyperparameters
    # parser.add_argument('--batch_size', type=int, default=-1)  # 自動隨著portion決定!
    parser.add_argument('--lr', type=float, default=5e-5)  
    parser.add_argument('--weight_decay', type=float)    
    
    return parser.parse_args()

def initialize_model(num_classes, pretrained, learning_rate):
    model = get_resnet18_classifier(num_classes = num_classes, pretrained = pretrained)
    criterion = nn.CrossEntropyLoss()
    optimizer_ = optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler_ = lr_scheduler.StepLR(optimizer_, step_size=5, gamma=0.1)

    return model, criterion, optimizer_, lr_scheduler_


def initialize_simclr_model(num_classes, learning_rate, weight_decay, simclr_path):
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

def main():

    args = parse_arguments()
    random.seed(args.seed)
    
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

    file_name = f"{args.AL_strategy}_{args.seed}_portion_{args.portion_start}_{args.portion_end}_{args.portion_interval}"
    print(f'Exp name: {file_name}')

    file_name = f"{file_name}_lr{args.lr}.json"

    
    ## Get Initial Label/Unlabel Idx
    tot_num_train = get_num_train(data_dir)
    print(f'Total Number of Train: {tot_num_train}')
    label_idx = []
    unlabeled_idx = list(range(tot_num_train))

    ## Train Model
    for portion in np.arange(args.portion_start, args.portion_end, args.portion_interval):
        if portion <= 20:
            batch_size = 2
        elif portion <= 60:
            batch_size = 4
        else:
            batch_size = 8
        
        #### First Iteration
        if portion == args.portion_start:
            if args.load_first_model:
                print('Load trained model as first iteration model!')
                model, criterion, optimizer_, lr_scheduler_ = initialize_simclr_model(num_classes, args.lr, args.weight_decay, args.simclr_path)
                state_dict = torch.load(args.first_model_path, map_location=args.device)
                last_trained_model.load_state_dict(state_dict)
                to_label_idx = [135, 141, 353, 369, 738, 942, 997, 1187, 1557, 1961]
                label_idx.extend(to_label_idx)
                unlabeled_idx = list(set(unlabeled_idx) - set(to_label_idx))
                print("len_label:", len(label_idx), "| len_unlabel:" , len(unlabeled_idx))
                continue # do not execute remaining code in current for loop
        
            ## Initialize Model
            if args.pretrained_weights == 'simclr':
                print('initialize resnet18 using simclr weights!')
                model, criterion, optimizer_, lr_scheduler_ = initialize_simclr_model(num_classes, args.lr, args.weight_decay, args.simclr_path)
            else:
                print('ImageNet Pretrained!')
                model, criterion, optimizer_, lr_scheduler_ = initialize_model(num_classes, True, args.lr) 

            ## Load Data
            print(f'===== Load {portion}% Data =====')
            target_num = round(tot_num_train * portion / 100)
            num_to_label = target_num - len(label_idx)
            random.seed(args.seed)
            print(to_label_idx)
            print("len_label_to_add:", num_to_label)
            if len(to_label_idx) != num_to_label:
                raise ValueError()
            label_idx.extend(to_label_idx)
            unlabeled_idx = list(set(unlabeled_idx) - set(to_label_idx))
            print("len_label:", len(label_idx), "| len_unlabel:" , len(unlabeled_idx))
            print(f'Batch Size: {batch_size}')
            data_loaders, dataset_sizes = get_data(data_dir, label_idx, batch_size)
            
            ## Train Model 
            print(f'===== Train Model with {portion}% data=====')
            last_trained_model, final_acc = train_model(model, args.device, data_loaders, dataset_sizes, criterion, optimizer_, lr_scheduler_, num_epochs=args.epoch)
            print("Final Acc: ", final_acc)


        #### Remaining Iteration
        else:
            ## Initialize Model
            if args.pretrained_weights == 'simclr':
                print('initialize resnet18 using simclr weights!')
                model, criterion, optimizer_, lr_scheduler_ = initialize_simclr_model(num_classes, args.lr, args.weight_decay, args.simclr_path)
            else:
                print('No Pretrained weights!')
                model, criterion, optimizer_, lr_scheduler_ = initialize_model(num_classes, True, args.lr) ## 現在先用ImageNet Pretrained --> 挖靠! 這裡要改欸!

            ## Load Data
            target_num = round(tot_num_train * portion / 100)
            num_to_label = target_num - len(label_idx)
            print(f'===== Load {portion}% Data =====')

            ### Select Images to Label
            if args.AL_strategy == 'random':
                # random.seed(args.seed)
                to_label_idx = random.sample(unlabeled_idx, num_to_label)
            elif args.AL_strategy == 'conf':
                if args.skip_top_k is not None:
                    to_label_idx, _ = conf(last_trained_model, data_dir, unlabeled_idx, len(unlabeled_idx), args.device)
                    num_skip = round(args.skip_top_k * len(unlabeled_idx) / 100)
                    print(f'skipping top {args.skip_top_k}% ({num_skip}) of unlabeld data')
                    to_label_idx = to_label_idx[num_skip: (num_skip + num_to_label)]
                else:
                    to_label_idx, _ = conf(last_trained_model, data_dir, unlabeled_idx, num_to_label, args.device)
            elif args.AL_strategy == 'random_conf':
                random.seed(42)
                to_label_idx, _ = conf(last_trained_model, data_dir, unlabeled_idx, len(unlabeled_idx), args.device)
                candidate_pool = to_label_idx[:round(3 * num_to_label)]
                to_label_idx = random.sample(candidate_pool, num_to_label)
            elif args.AL_strategy in ['density_conf', 'reverse_density_conf']:
                to_label_idx, _ = conf(last_trained_model, data_dir, unlabeled_idx, len(unlabeled_idx), args.device)
                candidate_pool = to_label_idx[:round(1.5 * num_to_label)]  # 用前 1.5 倍的 top uncertainty
                print('embed all train images using current model')
                backbone = last_trained_model.backbone
                modules = list(backbone.children())[:-1]
                feature_extractor = nn.Sequential(*modules)  # 512 dims
                train_feat_dict = get_latent_features(
                    data_dir, list(range(tot_num_train)),
                    extractor=feature_extractor,
                    device=args.device,
                    split='train',
                    pass_model=True
                )
                print('calculate density for candidate images')
                density_scores = compute_density_scores(candidate_pool, train_feat_dict, k=10, device=args.device)
                # 根據 density 分數排序後取前 num_to_label 個樣本
                if args.AL_strategy == 'density_conf':
                    sorted_density = sorted(density_scores.items(), key=lambda x: -x[1])
                else:
                    sorted_density = sorted(density_scores.items(), key=lambda x: x[1])
                to_label_idx = [idx for idx, _ in sorted_density[:num_to_label]]
            elif args.AL_strategy == 'reverse_conf':
                print('reverse conf!!')
                to_label_idx, _ = conf(last_trained_model, data_dir, unlabeled_idx, len(unlabeled_idx), args.device)
                to_label_idx = to_label_idx[-num_to_label:]
            elif args.AL_strategy == 'middle_conf':
                print('middle conf!!')
                to_label_idx, _= conf(last_trained_model, data_dir, unlabeled_idx, len(unlabeled_idx), args.device)
                # 從中間取 num_to_label 個元素
                mid = len(to_label_idx) // 2
                half_num = num_to_label // 2
                to_label_idx = to_label_idx[mid - half_num:mid - half_num + num_to_label]
            elif args.AL_strategy == 'margin':
                to_label_idx = margin(last_trained_model, data_dir, unlabeled_idx, num_to_label, args.device)
            elif args.AL_strategy == 'entropy':
                to_label_idx = entropy(last_trained_model, data_dir, unlabeled_idx, num_to_label, args.device)    
            elif args.AL_strategy == 'coreset':
                to_label_idx = coreset(last_trained_model, data_dir, unlabeled_idx, num_to_label, args.device)
            elif args.AL_strategy == 'badge':
                to_label_idx = badge(last_trained_model, data_dir, unlabeled_idx, num_to_label, args.device)
            else:
                raise NotImplementedError()
            

            print("len_label_to_add:", len(to_label_idx))
            label_idx.extend(to_label_idx)
            unlabeled_idx = list(set(unlabeled_idx) - set(to_label_idx))
            print("len_label:", len(label_idx), "| len_unlabel:" , len(unlabeled_idx))
            print(f'Batch Size: {batch_size}')
            data_loaders, dataset_sizes = get_data(data_dir, label_idx, batch_size)
            
            has_dup = len(label_idx) != len(set(label_idx))
            has_dup_2 = len(unlabeled_idx) != len(set(unlabeled_idx))
            if has_dup or has_dup_2:
                raise ValueError()

            ## Train Model 
            print(f'===== Train Model with {portion}% data=====')
            # if args.AL_add_on is None:
            if args.sam:
                print('Training using SAM')
                last_trained_model, final_acc = train_model_w_sam(model, args.device, data_loaders, dataset_sizes, criterion, optimizer_, lr_scheduler_, num_epochs=args.epoch)
            else:
                last_trained_model, final_acc = train_model(model, args.device, data_loaders, dataset_sizes, criterion, optimizer_, lr_scheduler_, num_epochs=args.epoch)
                
            if args.AL_add_on == 'trustAL':
                raise NotImplementedError
            print("Final Acc: ", final_acc)

        ## Save Results
        save_path = os.path.join(args.exp_path, f"iterative_pretrained_{args.pretrained_weights}", f"classification_{args.task_type}")
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, file_name)

        # 初始化文件（如果不存在）
        if not os.path.isfile(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("{}")

        # 讀取現有數據並更新
        with open(file_path, "r+", encoding="utf-8") as f:
            data = json.load(f)
            
            # 添加新數據，結構為 {"acc": accuracy, "label_idx": [list_of_indices]}
            data[str(float(portion))] = {
                "acc": final_acc,
                "label_idx": label_idx
            }
            
            # 按portion值排序
            sorted_data = {k: data[k] for k in sorted(data, key=lambda x: float(x))}
            
            # 寫回文件 - 使用自定義格式化
            f.seek(0)
            f.truncate()
            
            # 自定義JSON格式化，讓label_idx保持在一行
            json_str = "{\n"
            for i, (key, value) in enumerate(sorted_data.items()):
                json_str += f'    "{key}": {{\n'
                json_str += f'        "acc": {value["acc"]},\n'
                json_str += f'        "label_idx": {json.dumps(value["label_idx"])}\n'
                json_str += "    }"
                if i < len(sorted_data) - 1:
                    json_str += ","
                json_str += "\n"
            json_str += "}"
            
            f.write(json_str)

        print('result saved!')
    return

if __name__ == "__main__":
    main()