import os
import sys
print(f"Current working directory: {os.getcwd()}")
sys.path.insert(0, os.getcwd())

import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from classification.utils.data import get_data, get_num_train
from classification.model.resnet import get_resnet18_classifier
from classification.model.simclr.resnet_simclr import ResNetSimCLR 
from classification.utils.train_eval import train_model


def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, choices=['easy', 'medium', 'hard'], required=True)
    parser.add_argument('--portion', type=float, required=True) 
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')   
    parser.add_argument('--exp_path', type=str, default='./exp_results')   
    parser.add_argument('--epoch', type=int, default=20)   
    parser.add_argument('--lr', type=float, default=5e-5) 
    parser.add_argument('--weight_decay', type=float, default=None)   
    parser.add_argument('--no_use_pretrained', dest='use_pretrained', action='store_false', default=True)
    parser.add_argument('--pretrained_weights', type=str, choices=['simclr', 'auto_encoder'], default=None)   
    parser.add_argument('--simclr_path', type=str, default=None)
    parser.add_argument('--ask_saving', action='store_true', default=False)
    parser.add_argument('--no_data_aug', dest='data_aug', action='store_false', default=True)
    parser.add_argument('--aug_factor', type=int, default=4)   
    parser.add_argument('--flip_type', type=str, default='horizontal')   

    return parser.parse_args()


def initialize_model(num_classes, pretrained):
    model = get_resnet18_classifier(num_classes=num_classes, pretrained=pretrained)
    return model


def initialize_simclr_model(num_classes, simclr_path):
    model = ResNetSimCLR('resnet18', 32)
    state_dict = torch.load(simclr_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    
    in_features = model.backbone.fc[0].in_features
    model.backbone.fc = nn.Linear(in_features, num_classes, bias=True)
    return model

# def initialize_ae_model(num_classes, ae_path):
#     from SSL.auto_encoder.classes.resnet_autoencoder import AE
    
#     ae = AE('default')
#     ckpt = torch.load('./SSL/auto_encoder/autoencoder_1.ckpt', map_location='cpu')
#     state_dict = ckpt.get('model_state_dict', ckpt)
#     ae.load_state_dict(state_dict, strict=False)
    
#     encoder = ae.encoder
#     model = nn.Sequential(
#         encoder,
#         nn.AdaptiveAvgPool2d((1, 1)),
#         nn.Flatten(),
#         nn.Linear(512, num_classes, bias=True)
#     )
#     return model


def main():
    args = parse_arguments()
    
    # Determine task configuration
    task_config = {
        'easy': (2, './ds/classification/two_class'),
        'medium': (4, './ds/classification/four_class'),
        'hard': (7, './ds/classification/seven_class')
    }
    num_classes, data_dir = task_config[args.task_type]
    
    # Auto-adjust batch size
    # if args.portion <= 20:
    #     args.batch_size = 2
    # elif args.portion <= 60:
    #     args.batch_size = 4
    # else:
    #     args.batch_size = 8
    args.batch_size = 16
    
    # Generate file name
    file_name = f"random{args.seed}_lr{args.lr}_bs{args.batch_size}"
    if args.weight_decay:
        file_name += f"_wd{args.weight_decay}"
    file_name += ".json"
    
    print(f'Exp name: {file_name}')
    
    # Get training data indices
    tot_num_train = get_num_train(data_dir)
    print(f'Total Number of Train: {tot_num_train}')
    
    print(f'===== Load {args.portion}% Data =====')
    target_num = round(tot_num_train * args.portion / 100)
    
    # Sample indices
    random.seed(args.seed)
    unlabeled_idx = list(range(tot_num_train))
    label_idx = random.sample(unlabeled_idx, target_num)
    
    print(f"Number of labeled samples: {len(label_idx)}")
    # print(f"Label indices: {label_idx}")
    
    # Load data
    if args.data_aug:
        if args.aug_factor is None:
            raise ValueError()
        else:
            data_loaders, dataset_sizes = get_data(data_dir, label_idx, args.batch_size, data_aug=True, aug_factor=args.aug_factor, flip_type=args.flip_type)
    else:
        data_loaders, dataset_sizes = get_data(data_dir, label_idx, args.batch_size, data_aug=False)

    print(dataset_sizes)
    
    # Initialize model
    if args.use_pretrained:
        if args.pretrained_weights == 'simclr':
            print('Initialize ResNet18 using SimCLR weights')
            model = initialize_simclr_model(
                num_classes, args.simclr_path
            )
        elif args.pretrained_weights == 'auto_encoder':
            raise NotImplementedError()
            # print('Initialize ResNet18 using AutoEncoder weights')
            # model = initialize_ae_model(
            #     num_classes, args.ae_path
            # )
        else:
            print('Initialize ResNet18 using ImageNet pretrained weights')
            model = initialize_model(
                num_classes, True
            )
    else:
        print('Initialize ResNet18 without pretrained weights')
        model = initialize_model(
            num_classes, False
        )
    
    criterion = nn.CrossEntropyLoss()
    # if args.weight_decay is not None:
    #     print(f"Set weight decay to {args.weight_decay}")
    #     optimizer_ = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # else:
    #     optimizer_ = optim.Adam(model.parameters(), lr=args.lr)
    if args.weight_decay is not None:
        print(f"Set weight decay to {args.weight_decay}")
        optimizer_ = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer_ = optim.AdamW(model.parameters(), lr=args.lr)
    # lr_scheduler_ = lr_scheduler.StepLR(optimizer_, step_size=5, gamma=0.1)
    lr_scheduler_ = lr_scheduler.LinearLR(
        optimizer_, 
        start_factor=1.0,      # 從 100% 的 lr 開始
        end_factor=0.0,        # 降到 0% 的 lr（或設定其他值如 0.01）
        total_iters=args.epoch
    )
    
    # Train model
    print(f'===== Train Model with {args.portion}% data =====')
    print('-' * 50)
    print(f'{"Batch Size":<20}: {args.batch_size}')
    print(f'{"Learning Rate":<20}: {args.lr}')
    print(f'{"Weight Decay":<20}: {args.weight_decay if args.weight_decay else "None"}')
    print(f'{"Total Samples":<20}: {len(label_idx)}')
    print(f'{"Batches per Epoch":<20}: {len(label_idx) // args.batch_size}')

    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{"Trainable Params":<20}: {total_trainable_params / 1e6:.2f}M')
    print('-' * 50)


    _, final_acc = train_model(
        model, args.device, data_loaders, dataset_sizes, 
        criterion, optimizer_, lr_scheduler_, num_epochs=args.epoch
    )
    print(f"Final Acc: {final_acc}")
    
    # Optional save model
    if args.ask_saving:
        user_input = input('Do you want to save the model checkpoint? (y/n): ').strip().lower()
        if user_input in ['y', 'yes']:
            model_path = input('Please enter the path to save the model: ').strip()
            
            if not model_path:
                print('No path provided. Using default: ./model_checkpoint.pth')
                model_path = './model_checkpoint.pth'
            
            model_dir = os.path.dirname(model_path)
            if model_dir:
                os.makedirs(model_dir, exist_ok=True)
            
            try:
                torch.save(model.state_dict(), model_path)            
                print(f'Model checkpoint saved to: {model_path}')
            except Exception as e:
                print(f'Error saving model: {e}')
        else:
            print('Model checkpoint not saved.')
    
    # Save results
    if args.use_pretrained and args.pretrained_weights:
        save_path = os.path.join(args.exp_path, f"classification_{args.task_type}", 
                                f"cold_start_pretrained_{args.pretrained_weights}")
    elif args.use_pretrained:
        save_path = os.path.join(args.exp_path, f"classification_{args.task_type}", "cold_start")
    else:
        save_path = os.path.join(args.exp_path, f"classification_{args.task_type}", "cold_start_no_pretrained")
    
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, file_name)
    
    # Load or create results file
    if os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    
    # Update results
    portion_key = str(float(args.portion))
    if portion_key not in data:
        data[portion_key] = []
    data[portion_key].append(final_acc)
    
    # Sort and save
    sorted_data = {k: data[k] for k in sorted(data, key=float)}
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, indent=4, ensure_ascii=False)
    
    print(f'Result saved to {file_path}!')


if __name__ == "__main__":
    main()