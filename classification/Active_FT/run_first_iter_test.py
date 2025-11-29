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
from Classification.model import get_resnet18_classifier, train_model, train_model_w_sam, train_model_w_trustAL, eval_model
from AL_strategy.uncertainty import conf
from AL_strategy.diversity import coreset
from AL_strategy.hybrid import badge

# from feature_extract.util import get_latent_features, get_latent_features_vit, k_means_centroid, k_means_dense_center, k_means_cluster_margin
from feature_extract.util import *
from feature_extract.medimage_insight import get_latent_features_med



from Classification.Active_FT.ActiveFT import get_sample_ids


from model_merging.simple_avg import simple_avg

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, choices=['easy', 'medium', 'hard'], help='')
    parser.add_argument('--portion', type=float) 
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', type=str, default='cuda:0')   
    parser.add_argument('--exp_path', type=str, default='./Classification/Active_FT/exp_results')   
    parser.add_argument('--epoch', type=int, default=20)   

    parser.add_argument('--batch_size', type=int, default=8)   
    parser.add_argument('--lr', type=float, default=5e-5)   

    ## pretrained_related --> default to using weights pretrained on ImageNet
    parser.add_argument(
        '--no_use_pretrained',
        dest='use_pretrained',
        action='store_false',
        default=True,
        help='Disable loading of pretrained weights'
    )
    parser.add_argument('--pretrained_weights', type=str, choices=['simclr', 'auto_encoder'])   
    
    
    # cold start related
    # parser.add_argument('--label_idx_path', type=str, default='./exp_results')   
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
                                                          'MedImageInsight', 
                                                          'clip_base_32', 'clip_base_16', 'clip_large_14', 'clip_large_14_336' 
                                                          ], 
                                                          help='')

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
    label_idx = []
    unlabeled_idx = list(range(tot_num_train))

    ## Load Data
    print(f'===== Load {args.portion}% Data =====')
    target_num = round(tot_num_train * args.portion / 100)
    num_to_label = target_num - len(label_idx)
    
    distance = 'cosine'
    balance = 3.0
    to_label_idx = get_sample_ids(
        feature_path=f'./Classification/Active_FT/features_{args.extractor}.npy',
        percent=args.portion,
        temperature=0.07,
        lr=0.001,
        # lr=8e-4,
        max_iter=300,
        init='fps',
        distance=distance, 
        balance=balance
    )
    print("len_label_to_add:", num_to_label)
    print(to_label_idx)
    
    ## Train Model
    #### First Iteration only    
    
    label_idx.extend(to_label_idx)
    unlabeled_idx = list(set(unlabeled_idx) - set(to_label_idx))
    print("len_label:", len(label_idx), "| len_unlabel:" , len(unlabeled_idx))
    data_loaders, dataset_sizes = get_data(data_dir, label_idx, args.batch_size)
    print(dataset_sizes)
    
    ## Train Model 
    if args.model_merging == False:
        print(f'===== Train Model with {args.portion}% data=====')
        ## Initialize Model
        if args.use_pretrained:
            if args.pretrained_weights == 'simclr':
                print('initialize resnet18 using simclr weights!')
                model, criterion, optimizer_, lr_scheduler_ = initialize_simclr_model(num_classes, args.lr)
            else:
                print('initialize resnet18 using pretrained weights (i.e. ImageNet1K)!')
                model, criterion, optimizer_, lr_scheduler_ = initialize_model(num_classes, True, args.lr)
        else:
            print('initialize resnet18 with no pretrained weights!')
            model, criterion, optimizer_, lr_scheduler_ = initialize_model(num_classes, False, args.lr)
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_trainable_params}")

        _, final_acc = train_model(model, args.device, data_loaders, dataset_sizes, criterion, optimizer_, lr_scheduler_, num_epochs=args.epoch)
        
        print("Final Acc: ", final_acc)
    else:
        trained_models = []
        for i in range(args.n_models):
            print(f'Start training model {i+1}')
            print(f'===== Train Model with {args.portion}% data=====')
            ## Initialize Model
            if args.use_pretrained:
                if args.pretrained_weights == 'simclr':
                    print('initialize resnet18 using simclr weights!')
                    model, criterion, optimizer_, lr_scheduler_ = initialize_simclr_model(num_classes)
                elif args.pretrained_weights == 'auto_encoder':
                    model, criterion, optimizer_, lr_scheduler_ = initialize_ae_model(num_classes)
                else:
                    print('initialize resnet18 using pretrained weights (i.e. ImageNet1K)!')
                    model, criterion, optimizer_, lr_scheduler_ = initialize_model(num_classes, True)
            else:
                print('initialize resnet18 with no pretrained weights!')
                model, criterion, optimizer_, lr_scheduler_ = initialize_model(num_classes, False)
            total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total trainable parameters: {total_trainable_params}")

            trained_model, final_acc = train_model(model, args.device, data_loaders, dataset_sizes, criterion, optimizer_, lr_scheduler_, num_epochs=args.epoch)
            print("Final Acc: ", final_acc)
            
            trained_models.append(trained_model)
        print('model merging')
        merged_model = simple_avg(trained_models)
        eval_loss, final_acc = eval_model(merged_model, args.device, data_loaders['val'], dataset_sizes['val'], criterion)
        print("eval loss:", eval_loss, "Final Acc: ", final_acc)

    if args.use_pretrained:
        file_path = os.path.join(args.exp_path, f"{args.pretrained_weights}_{distance}_balance{balance}", f"{args.extractor}_bs{args.batch_size}.json")
    else:
        file_path = os.path.join(args.exp_path, f"no_pretrained_{distance}_balance{balance}", f"{args.extractor}_bs{args.batch_size}.json")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
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